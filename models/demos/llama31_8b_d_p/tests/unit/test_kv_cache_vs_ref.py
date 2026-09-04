# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`tt/attention/kv_cache.py` — the cache **primitive**. Gate: `G-KV`.

The KV cache is the output of prefill, so its correctness is the whole point. This gate proves the
primitive: `allocate_kv_cache` builds the right geometry at the real `head_dim = 128`,
`write_kv_chunk` puts each token at the right global position, and it writes **nothing else**.

* **What it deliberately does not prove: the model -> cache path.** At `(1,1)` the model emits all
  8 KV heads on one chip while the per-chip cache holds exactly **one**, and the write op refuses
  the mismatch outright
  (`TT_FATAL: cache and input num-heads dim must match`,
  `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp:230`).
  So this gate drives the op **one head at a time**, using the cache's own layer slots.
  `G-KV-TP8` (P8) owns the model -> cache path; `07_RISKS.md` R-001 carries the gap. Also, at
  `sp = 1` the block-cyclic layout **degenerates to the identity**, so nothing here tests the
  block-cyclic reorder either.
* **Input distribution:** for the PCC half, realistic **post-RoPE K** and **raw V** — standard-normal
  `x`, `randn * 0.02` projections, the package's own llama3 RoPE tables — because a cache round trip
  should be measured on the values it will actually hold. For the bit-exact half, an integer
  **position/head/chunk-labelled** payload; see below.
* **Reference dtype policy:** fp32 reference throughout; only the tensor the device *stores* is
  quantised for the floor.
* **Threshold:** PCC >= **0.99** at the cache dtype and **<= 3x its floor**
  (`BRINGUP_RECIPE.md:2071`); the bf16 number is recorded too, which is the delta `DEC-021` owes.
  Both should sit essentially **at** the floor, because the write is a copy, not arithmetic.
* **Every layout, mapping and address claim is gated on bit-equality** (`torch.equal`,
  `rtol=atol=0`), never PCC — recipe §2.5: a *rotated* head->column map still scored PCC 0.99890.
* **Three constraints on the positional probe. Two are §2.5's; the third corrects it.**
  1. **Several `kv_actual` offsets, not one** — 4 chunks landing at `kv_actual ∈ {0, 32, 64, 96}`.
  2. **Every 16-lane block carries a single constant value.** `bfloat8_b` shares one exponent per
     16-element block, so a block holding both `1` and `128` would round the `1` away.
  3. **Every encoded value stays <= 128 — not §2.5's 256.** §2.5 gives the ceiling as 256
     ("`bfloat16` is exact only up to 256"), which is right for bf16 and **wrong for the cache
     dtype the recipe itself mandates**. Measured on this box, with blocks already held constant:
     the first inexact integer is **129** at `bfloat8_b` and **257** at `bfloat16`. A probe built to
     the stated 256 fails at bf8_b on a perfectly correct cache — which is §2.5's own trap
     ("a failing probe is not evidence of a failing module until the probe's own numerics are
     checked") one dtype down. It fired here, in exactly that shape: chunk 2, odd rows,
     `max|delta| = 1.0`. `DEC-044`.

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_kv_cache_vs_ref.py -x -q
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama31_8b_d_p.tests.test_factory import err_ratio, llama_config_dims, quantize_like_device
from models.demos.llama31_8b_d_p.tt.attention.kv_cache import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    allocate_kv_cache,
    write_kv_chunk,
)
from models.demos.llama31_8b_d_p.tt.config import derive_head_dim
from models.demos.llama31_8b_d_p.tt.rope import llama3_freqs

PCC_THRESHOLD = 0.99  # `BRINGUP_RECIPE.md:2071`
MAX_ERR_RATIO = 3.0
WEIGHT_SCALE = 0.02

CACHE_DTYPES = [ttnn.bfloat8_b, ttnn.bfloat16]  # bf8_b is the deployment dtype (`DEC-021`)
_DTYPE_IDS = {ttnn.bfloat8_b: "bf8_b", ttnn.bfloat16: "bf16"}

# The positional probe's geometry. 4 x 32 = **128** positions, so every encoded id is in [0, 127],
# and the writes land at four different `kv_actual` offsets {0, 32, 64, 96}. `PROBE_MAX_SEQ_LEN`
# leaves a 256-position pad tail that must stay untouched; 384 is the length the recipe's own probe
# used.
#
# **128, not 256.** Recipe §2.5 gives the ceiling as 256 ("bfloat16 is exact only up to 256"), and
# that is right for bf16 — but the cache dtype is `bfloat8_b` (`DEC-021`), and bf8_b is exact only
# up to **128**: measured on this box, the first inexact integer is **129** at bf8_b and **257** at
# bf16, even with each 16-lane block held constant so the shared exponent is ideal. A probe built to
# the recipe's stated 256 fails at bf8_b on a perfectly correct cache — §2.5's own trap, one dtype
# down. `DEC-044`.
PROBE_CHUNK = 32
PROBE_CHUNKS = 4
PROBE_MAX_SEQ_LEN = 384
PROBE_LANE_BLOCK = 16  # `bfloat8_b`'s shared-exponent block; each block carries one constant value
# Per-dtype exact-integer ceiling, measured (not assumed) — see the note above.
EXACT_INTEGER_CEILING = {ttnn.bfloat8_b: 128, ttnn.bfloat16: 256}


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _reference_kv(hf, seq_len):
    """fp32 **post-RoPE K** and **raw V**, `[1, num_kv_heads, S, head_dim]` — what the cache holds.

    K post-RoPE and V raw is what every template stores
    (`models/demos/gpt_oss_d_p/tt/attention/prefill.py:162-165`); stating it matters because a cache
    holding pre-RoPE K would read back plausibly and attend wrongly.
    """
    nkv, head_dim, hidden = hf["num_key_value_heads"], derive_head_dim(hf), hf["hidden_size"]
    x = torch.randn(seq_len, hidden)
    k_w = torch.randn(nkv * head_dim, hidden) * WEIGHT_SCALE
    v_w = torch.randn(nkv * head_dim, hidden) * WEIGHT_SCALE

    def _proj(w):
        return (x @ w.transpose(-1, -2)).view(1, seq_len, nkv, head_dim).transpose(1, 2)

    cos_half, sin_half = llama3_freqs(hf, seq_len)
    cos = torch.cat([cos_half, cos_half], dim=-1)
    sin = torch.cat([sin_half, sin_half], dim=-1)
    k = _proj(k_w)
    return k * cos + _rotate_half(k) * sin, _proj(v_w)


def _positional_payload(chunk_idx, head_id, head_dim):
    """`[1, 1, PROBE_CHUNK, head_dim]` where **every row names its own global position**.

    Lane blocks, each `PROBE_LANE_BLOCK` wide and each holding a single constant so `bfloat8_b`'s
    shared exponent is exact:

    * block 0 -> the **global position** `chunk_idx * PROBE_CHUNK + row` (0..127)
    * block 1 -> `head_id + 1` (1-based, so it cannot be confused with a zeroed slot)
    * block 2 -> `chunk_idx + 1`
    * blocks 3+ -> the global position again

    Every value is an integer in `[0, 127]` — at or below the **measured** bf8_b exact-integer
    ceiling of 128, not §2.5's bf16 ceiling of 256 (`DEC-044`) — and constant within each 16-lane
    block, so `bfloat8_b`'s shared exponent is exact as well.
    """
    assert head_dim % PROBE_LANE_BLOCK == 0
    positions = torch.arange(chunk_idx * PROBE_CHUNK, (chunk_idx + 1) * PROBE_CHUNK, dtype=torch.float32)
    assert positions.max() <= min(EXACT_INTEGER_CEILING.values()), (
        f"probe payload reaches {int(positions.max())}, above the bf8_b exact-integer ceiling "
        f"{EXACT_INTEGER_CEILING[ttnn.bfloat8_b]} — the probe would fail on a correct cache (DEC-044)"
    )
    blocks = []
    for block_idx in range(head_dim // PROBE_LANE_BLOCK):
        if block_idx == 1:
            value = torch.full_like(positions, float(head_id + 1))
        elif block_idx == 2:
            value = torch.full_like(positions, float(chunk_idx + 1))
        else:
            value = positions
        blocks.append(value[:, None].expand(PROBE_CHUNK, PROBE_LANE_BLOCK))
    return torch.cat(blocks, dim=-1)[None, None]


def _to_device(t, mesh_device, dtype):
    return ttnn.from_torch(
        t,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _readback(cache):
    """The single chip's whole cache tensor, `[num_users*num_layers, 1, seq_local, head_dim]`, fp32.

    At `sp = 1` the block-cyclic layout is the identity, so local row == global position and no
    inverse reorder is needed. That is also exactly why this gate cannot test the reorder.
    """
    return ttnn.to_torch(ttnn.get_device_tensors(cache)[0]).float()


def _slot(kv_cache, user_id, layer_idx):
    return user_id * kv_cache.num_layers + layer_idx


# --------------------------------------------------------------------------------------------
# The PCC half: realistic post-RoPE K / raw V through a write + read-back round trip.
# --------------------------------------------------------------------------------------------
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES, ids=lambda d: _DTYPE_IDS[d])
@pytest.mark.parametrize("seq_len", [128, 512], ids=lambda s: f"s{s}")
def test_kv_cache_write_vs_ref(mesh_device, cache_dtype, seq_len, reset_seeds):
    """Write each KV head's post-RoPE K and raw V, read it back, PCC vs the fp32 reference.

    One head per call, into its own **layer slot**, because the per-chip cache holds one KV head —
    which also exercises `layer_idx` indexing rather than only slot 0.
    """
    hf = llama_config_dims()
    nkv, head_dim = hf["num_key_value_heads"], derive_head_dim(hf)
    ref_k, ref_v = _reference_kv(hf, seq_len)

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=nkv,  # one layer slot per synthetic head
        max_seq_len=seq_len,
        num_users=1,
        head_dim=head_dim,
        cache_dtype=cache_dtype,
    )
    assert kv_cache.sp == 1 and kv_cache.max_seq_len == seq_len

    for head in range(nkv):
        write_kv_chunk(
            kv_cache,
            _to_device(ref_k[:, head : head + 1], mesh_device, cache_dtype),
            _to_device(ref_v[:, head : head + 1], mesh_device, cache_dtype),
            slot_idx=0,
            layer_idx=head,
            kv_actual=0,
            sp_axis=0,
        )
    ttnn.synchronize_device(mesh_device)

    host_k, host_v = _readback(kv_cache.k), _readback(kv_cache.v)

    worst = {}
    for name, ref, host in (("K", ref_k, host_k), ("V", ref_v, host_v)):
        for head in range(nkv):
            ref_head = ref[:, head : head + 1]
            got = host[_slot(kv_cache, 0, head) : _slot(kv_cache, 0, head) + 1].unsqueeze(0)[..., :seq_len, :]
            _, floor = comp_pcc(ref_head, quantize_like_device(ref_head, cache_dtype), 0.0)
            passing, pcc = comp_pcc(ref_head, got.reshape(ref_head.shape), PCC_THRESHOLD)
            ratio = err_ratio(float(pcc), float(floor))
            assert passing, f"{name} head {head} below {PCC_THRESHOLD}: {pcc}"
            assert ratio <= MAX_ERR_RATIO, f"{name} head {head} is {ratio:.2f}x off its floor"
            if name not in worst or float(pcc) < worst[name][0]:
                worst[name] = (float(pcc), float(floor), ratio, head)

    logger.info(
        f"[G-KV] round trip {_DTYPE_IDS[cache_dtype]} seq={seq_len}: worst of {nkv} heads — "
        f"K PCC={worst['K'][0]:.7f} floor={worst['K'][1]:.7f} ratio={worst['K'][2]:.2f}x (head {worst['K'][3]}); "
        f"V PCC={worst['V'][0]:.7f} floor={worst['V'][1]:.7f} ratio={worst['V'][2]:.2f}x (head {worst['V'][3]})"
    )


# --------------------------------------------------------------------------------------------
# The bit-exact half: the position map, and that a later write leaves earlier chunks alone.
# --------------------------------------------------------------------------------------------
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES, ids=lambda d: _DTYPE_IDS[d])
def test_kv_cache_positional_readback_bit_exact(mesh_device, cache_dtype, reset_seeds):
    """Every row must land at its own global position, **bit-exactly** (`torch.equal`).

    An address claim gated on PCC is not gated: recipe §2.5 measured a *rotated* head->column map
    still scoring 0.99890. This asserts `rtol=atol=0` over four chunks at four different `kv_actual`
    offsets, and after each chunk re-checks that **every earlier chunk is still unchanged**.
    """
    hf = llama_config_dims()
    head_dim = derive_head_dim(hf)
    head_id = 3  # an arbitrary non-zero head, so a head-id lane of 0 would be noticed

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=PROBE_MAX_SEQ_LEN,
        num_users=1,
        head_dim=head_dim,
        cache_dtype=cache_dtype,
    )

    for chunk_idx in range(PROBE_CHUNKS):
        payload = _positional_payload(chunk_idx, head_id, head_dim)
        kv_actual = chunk_idx * PROBE_CHUNK
        assert kv_actual % ttnn.TILE_SIZE == 0
        write_kv_chunk(
            kv_cache,
            _to_device(payload, mesh_device, cache_dtype),
            _to_device(payload, mesh_device, cache_dtype),
            slot_idx=0,
            layer_idx=0,
            kv_actual=kv_actual,
            sp_axis=0,
        )
        ttnn.synchronize_device(mesh_device)

        host = _readback(kv_cache.k)[0, 0]  # [PROBE_MAX_SEQ_LEN, head_dim]
        # Every chunk written SO FAR must still be exactly its own payload — which is both the
        # position-map assertion and "an earlier chunk unchanged after a later chunk's write".
        for earlier in range(chunk_idx + 1):
            expected = _positional_payload(earlier, head_id, head_dim)[0, 0]
            got = host[earlier * PROBE_CHUNK : (earlier + 1) * PROBE_CHUNK]
            assert torch.equal(got, expected), (
                f"chunk {earlier} is not bit-identical after writing chunk {chunk_idx} "
                f"(max|delta| = {float((got - expected).abs().max())}); "
                f"rows {torch.nonzero((got != expected).any(dim=-1)).flatten().tolist()[:8]}"
            )

    written = PROBE_CHUNKS * PROBE_CHUNK
    logger.info(
        f"[G-KV] positional read-back {_DTYPE_IDS[cache_dtype]}: {written} rows across "
        f"{PROBE_CHUNKS} chunks at kv_actual {[c * PROBE_CHUNK for c in range(PROBE_CHUNKS)]} — "
        f"bit-identical (torch.equal, rtol=atol=0), max encoded value "
        f"{int(_positional_payload(PROBE_CHUNKS - 1, head_id, head_dim).max())} <= "
        f"{EXACT_INTEGER_CEILING[cache_dtype]} (the measured exact-integer ceiling for this dtype)"
    )


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("cache_dtype", CACHE_DTYPES, ids=lambda d: _DTYPE_IDS[d])
def test_kv_cache_no_collateral_writes(mesh_device, cache_dtype, reset_seeds):
    """The written region **only**: pad tail untouched, every other (user, layer) exactly zero.

    All three checks are bit-exact. A write that leaked into another user's slot would be an
    out-of-bounds serving bug, not a numerical one, so PCC is the wrong instrument (§2.5).
    """
    hf = llama_config_dims()
    head_dim = derive_head_dim(hf)
    num_users, num_layers = 2, 2
    target_user, target_layer = 0, 1  # deliberately not (0, 0), so a slot-index bug shows up

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=PROBE_MAX_SEQ_LEN,
        num_users=num_users,
        head_dim=head_dim,
        cache_dtype=cache_dtype,
    )

    payload = _positional_payload(0, 0, head_dim)
    write_kv_chunk(
        kv_cache,
        _to_device(payload, mesh_device, cache_dtype),
        _to_device(payload, mesh_device, cache_dtype),
        slot_idx=target_user,
        layer_idx=target_layer,
        kv_actual=0,
        sp_axis=0,
    )
    ttnn.synchronize_device(mesh_device)

    for name, cache in (("K", kv_cache.k), ("V", kv_cache.v)):
        host = _readback(cache)
        target = _slot(kv_cache, target_user, target_layer)

        assert torch.equal(host[target, 0, :PROBE_CHUNK], payload[0, 0]), f"{name}: the written rows are wrong"

        pad_tail = host[target, 0, PROBE_CHUNK:]
        assert torch.equal(pad_tail, torch.zeros_like(pad_tail)), (
            f"{name}: the pad tail was written — max|value| = {float(pad_tail.abs().max())} over "
            f"{PROBE_MAX_SEQ_LEN - PROBE_CHUNK} positions past the chunk"
        )

        for user in range(num_users):
            for layer in range(num_layers):
                if (user, layer) == (target_user, target_layer):
                    continue
                other = host[_slot(kv_cache, user, layer)]
                assert torch.equal(other, torch.zeros_like(other)), (
                    f"{name}: writing (user {target_user}, layer {target_layer}) also touched "
                    f"(user {user}, layer {layer}) — max|value| = {float(other.abs().max())}"
                )

    logger.info(
        f"[G-KV] no collateral writes {_DTYPE_IDS[cache_dtype]}: written rows exact; pad tail "
        f"({PROBE_MAX_SEQ_LEN - PROBE_CHUNK} positions) exactly zero; the other "
        f"{num_users * num_layers - 1} (user, layer) slots exactly zero — all via torch.equal"
    )


# --------------------------------------------------------------------------------------------
# The dtype delta `DEC-021` owes: bf8_b vs bf16, measured rather than assumed.
# --------------------------------------------------------------------------------------------
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_kv_cache_dtype_delta(mesh_device, reset_seeds):
    """The PCC cost of `bfloat8_b` over `bfloat16` on real post-RoPE K / raw V (`DEC-021`).

    Recorded, not thresholded: `DEC-021` chose bf8_b on footprint and on matching the DeepSeek
    substrate, and what it owed `G-KV` was the **measured** cost of that choice.
    """
    hf = llama_config_dims()
    seq_len = 512
    ref_k, ref_v = _reference_kv(hf, seq_len)
    head_dim = derive_head_dim(hf)

    measured = {}
    for cache_dtype in CACHE_DTYPES:
        kv_cache = allocate_kv_cache(
            mesh_device, num_layers=1, max_seq_len=seq_len, num_users=1, head_dim=head_dim, cache_dtype=cache_dtype
        )
        write_kv_chunk(
            kv_cache,
            _to_device(ref_k[:, :1], mesh_device, cache_dtype),
            _to_device(ref_v[:, :1], mesh_device, cache_dtype),
            slot_idx=0,
            layer_idx=0,
            kv_actual=0,
            sp_axis=0,
        )
        ttnn.synchronize_device(mesh_device)
        _, pcc_k = comp_pcc(ref_k[:, :1], _readback(kv_cache.k)[:1].unsqueeze(0).reshape(ref_k[:, :1].shape), 0.0)
        _, pcc_v = comp_pcc(ref_v[:, :1], _readback(kv_cache.v)[:1].unsqueeze(0).reshape(ref_v[:, :1].shape), 0.0)
        measured[cache_dtype] = (float(pcc_k), float(pcc_v))

    k8, v8 = measured[ttnn.bfloat8_b]
    k16, v16 = measured[ttnn.bfloat16]
    logger.info(
        f"[G-KV] dtype delta (seq={seq_len}, head 0): bf8_b K={k8:.7f} V={v8:.7f}; "
        f"bf16 K={k16:.7f} V={v16:.7f}; cost of bf8_b = {(1 - k8) / (1 - k16):.1f}x on K, "
        f"{(1 - v8) / (1 - v16):.1f}x on V. Cache bytes per token per head: "
        f"{head_dim} at bf8_b vs {2 * head_dim} at bf16 (DEC-021)"
    )
    assert k8 >= PCC_THRESHOLD and v8 >= PCC_THRESHOLD, f"bf8_b below {PCC_THRESHOLD}: K={k8} V={v8}"
    assert k16 >= k8 and v16 >= v8, "bf16 is not better than bf8_b — recheck the round trip"


# --------------------------------------------------------------------------------------------
# Refusals: every one of them guards a silent-wrongness path.
# --------------------------------------------------------------------------------------------
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_kv_cache_refuses_batched_write(mesh_device, expect_error, reset_seeds):
    """`write_kv_chunk` writes ONE user per call and must say so.

    `update_padded_kv_cache` ignores the leading batch dim, so a `batch > 1` tensor would silently
    write only `slot_idx` and drop the rest
    (`models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:145-152`).
    """
    hf = llama_config_dims()
    head_dim = derive_head_dim(hf)
    kv_cache = allocate_kv_cache(mesh_device, num_layers=1, max_seq_len=64, num_users=2, head_dim=head_dim)
    batched = _to_device(torch.zeros(2, 1, 64, head_dim), mesh_device, ttnn.bfloat8_b)
    with expect_error(AssertionError, "one user per call"):
        write_kv_chunk(kv_cache, batched, batched, slot_idx=0, layer_idx=0, kv_actual=0, sp_axis=0)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "slot_idx, layer_idx, kv_actual, message",
    [
        (2, 0, 0, "slot_idx 2 out of range"),
        (0, 1, 0, "layer_idx 1 out of range"),
        (0, 0, 16, "must be tile-aligned"),
    ],
    ids=["slot_oob", "layer_oob", "misaligned_offset"],
)
def test_kv_cache_refuses_bad_write_target(mesh_device, expect_error, slot_idx, layer_idx, kv_actual, message):
    """A bad slot or layer is a silent OOB write into someone else's cache; a misaligned `kv_actual`
    breaks the block-cyclic per-device write. Both fail loud."""
    hf = llama_config_dims()
    head_dim = derive_head_dim(hf)
    kv_cache = allocate_kv_cache(mesh_device, num_layers=1, max_seq_len=64, num_users=2, head_dim=head_dim)
    chunk = _to_device(torch.zeros(1, 1, 32, head_dim), mesh_device, ttnn.bfloat8_b)
    with expect_error(AssertionError, message):
        write_kv_chunk(kv_cache, chunk, chunk, slot_idx=slot_idx, layer_idx=layer_idx, kv_actual=kv_actual, sp_axis=0)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_kv_cache_refuses_unaligned_capacity(mesh_device, expect_error):
    """`max_seq_len` must be a multiple of `TILE_SIZE * sp`, or `seq_local` is not tile-aligned."""
    # NOTE: `expect_error`'s `message` is matched as a **regex**, not as the substring its docstring
    # describes ("must appear in the real device error text"), so a message containing `*` — here
    # "multiple of TILE_SIZE*sp" — silently fails to match. Match on the metachar-free tail instead.
    with expect_error(AssertionError, "seq_local must be tile-aligned"):
        allocate_kv_cache(mesh_device, num_layers=1, max_seq_len=48, num_users=1, head_dim=128)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_kv_cache_geometry(mesh_device, reset_seeds):
    """The allocation's geometry, asserted exactly: shape, dtype, and the producer's block size.

    The `[1, 1, 32, head_dim]` DRAM shard is what lets P10 reuse the producer's existing packed-GQA
    read-back instead of writing a fourth reader (`BRINGUP_RECIPE.md:1456-1458`), so the constant is
    part of the contract rather than an implementation detail.
    """
    hf = llama_config_dims()
    head_dim = derive_head_dim(hf)
    num_users, num_layers, max_seq_len = 2, 32, PROBE_MAX_SEQ_LEN
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        num_users=num_users,
        head_dim=head_dim,
    )
    assert (
        NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK == 32
    ), "the producer's block geometry changed — see DEC-021's blast radius"
    for cache in (kv_cache.k, kv_cache.v):
        assert tuple(cache.shape) == (num_users * num_layers, 1, max_seq_len // kv_cache.sp, head_dim)
        assert cache.dtype == ttnn.bfloat8_b  # `DEC-021`
        assert cache.layout == ttnn.TILE_LAYOUT
    logger.info(
        f"[G-KV] geometry: per-chip {tuple(kv_cache.k.shape)} bf8_b TILE, "
        f"{num_users} users x {num_layers} layers = {num_users * num_layers} slots, "
        f"sp={kv_cache.sp}, DRAM shard [1, 1, {NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK}, {head_dim}]"
    )
