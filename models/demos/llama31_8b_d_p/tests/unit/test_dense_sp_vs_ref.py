# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The ring-joint SP attention core, **alone**, vs an fp32 torch reference. Gate: `G-SP-RING`.

**HF anchor:** `transformers.models.llama.modeling_llama.LlamaAttention`'s causal attention core.
The in-test reference is written here in ~20 lines (recipe P1 option 3), driven from the *same*
values the device holds, so no checkpoint is needed and the comparison cannot silently be against a
different tensor.

**What "alone" means and why it matters.** `tt/attention/dense_sp.py::dense_sp_attention` is called
with a cache this file populated itself and a Q this file built itself — no projections, no RoPE, no
`o_proj`, no TP collective. Recipe §2.3 is the reason: the fused kernel does not obey the floor
model, its slack must be **named and tracked separately** from the stages this package writes, and a
budget that lumps the two together can absorb a real regression. The single-card SDPA's standalone
probe measured **52.8-55.0x** its floor at `G-ATTN`; this file is the ring op's equivalent, and the
recipe's own figure for it is **7.98x** (`BRINGUP_RECIPE.md:1789-1791`).

## The geometry, and why the reference does not need to know the cache's internal layout

The cache is populated **through the real write op** (`write_kv_chunk` ->
`ttnn.experimental.deepseek_prefill.update_padded_kv_cache`) from a host tensor in plain
**global-position order**, mesh-mapped exactly the way the model maps a chunk: sequence sharded over
the SP rows, KV heads sharded over the TP columns (`ShardTensor2dMesh(dims=(2, 1))`). Chunk `c`'s
slice `[c*chunk_global, (c+1)*chunk_global)` therefore lands on device row `r` at local rows
`[c*chunk_local, (c+1)*chunk_local)`, which is the block-cyclic layout
(`tt/rope.py::build_indexed_rope` reorders the RoPE tables by the same period). Q is built the same
way, so **Q row `i` of device `r` is global position `kv_actual + r*chunk_local + i`** and the whole
comparison reduces to "causal attention, Q at global positions `[kv_actual, logical_n)`, K/V at
`[0, logical_n)`" — with no layout arithmetic in the reference at all.

Q head `h` attends KV head `h // 4` (GQA group 4), and the column split is group-consistent by
construction: TP column `c` holds Q heads `[4c, 4c+4)` and KV head `c`, and `4c // 4 == c` for every
one of the four. That is the same fact `G-ATTN` records for the dense op — the GQA group is native
to the kernel, there is no on-chip KV repeat.

Run:
    export TT_MESH_GRAPH_DESC_PATH=...   # only for PREFILL_TOPOLOGY=ring; see test_factory
    pytest models/demos/llama31_8b_d_p/tests/unit/test_dense_sp_vs_ref.py -x -q
"""

import math

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss_d_p.utils.general_utils import get_default_num_links
from models.demos.llama31_8b_d_p.tests.test_factory import (  # noqa: F401 — submesh_pool is a fixture
    GALAXY_MESH_SHAPE,
    err_ratio,
    galaxy_device_params,
    llama_config_dims,
    prefill_topology,
    quantize_like_device,
    requires_galaxy,
    requires_ring_fabric,
    submesh_pool,
)
from models.demos.llama31_8b_d_p.tt.attention.dense_sp import (
    dense_sp_attention,
    sp_ring_compute_kernel_config,
    sp_ring_program_config,
)
from models.demos.llama31_8b_d_p.tt.attention.kv_cache import allocate_kv_cache, write_kv_chunk
from models.demos.llama31_8b_d_p.tt.ccl import CCLManager
from models.demos.llama31_8b_d_p.tt.config import MeshConfig, derive_head_dim

ACTIVATION_DTYPE = ttnn.bfloat16  # `DEC-022` — Q comes out of the projection in bf16
CACHE_DTYPE = ttnn.bfloat8_b  # `DEC-021`

# `BRINGUP_RECIPE.md:1806-1808`: PCC >= 0.99 vs an fp32 torch reference, with the ratio to its own
# floor **reported** rather than asserted — §2.3 measures that a fused kernel does not sit at its
# floor and §2.3.1 that a ratio budget is not portable, so a threshold on this op's ratio would be
# a number invented here. The absolute 0.99 is the recipe's.
SP_RING_PCC_THRESHOLD = 0.99

# The shape. Chosen so the ring path is the one that runs and the SDPA chunking is legal:
#   chunk_global % (TILE_SIZE * sp) == 0        512 % 128 == 0
#   chunk_local == chunk_global // sp == 128 == the ring program config's q_chunk_size
#   max_seq_len > chunk_global                  2048 > 512, so `select_attention_core` -> sp_ring
#   kv_actual % TILE_SIZE == 0                  512
CHUNK_GLOBAL = 512
KV_ACTUAL = 512
CACHE_GLOBAL = 2048


def _torch_ring_reference(q, k, v, *, kv_actual, logical_n, gqa_group, scale):
    """fp32 causal attention: Q at global positions `[kv_actual, logical_n)`, K/V at `[0, logical_n)`.

    `q` `[1, n_q, chunk_global, head_dim]`, `k`/`v` `[1, n_kv, logical_n, head_dim]`, all in global
    position order. The causal mask is built **explicitly** — `torch.triu` of `-inf` offset by
    `kv_actual` — rather than relying on any op's default, which is the same discipline `G-ATTN`'s
    reference follows (recipe P5.5). KV heads are `repeat_interleave`d by the GQA group, which the
    device does **not** do.
    """
    n_q, chunk_global, head_dim = q.shape[1], q.shape[2], q.shape[3]
    k_rep = k.repeat_interleave(gqa_group, dim=1)
    v_rep = v.repeat_interleave(gqa_group, dim=1)
    scores = torch.matmul(q, k_rep.transpose(-1, -2)) * scale  # [1, n_q, chunk_global, logical_n]

    q_pos = torch.arange(kv_actual, kv_actual + chunk_global).unsqueeze(-1)  # [chunk_global, 1]
    k_pos = torch.arange(logical_n).unsqueeze(0)  # [1, logical_n]
    mask = torch.where(k_pos <= q_pos, 0.0, float("-inf"))
    scores = scores + mask
    return torch.matmul(torch.softmax(scores, dim=-1), v_rep), n_q, head_dim


def _shard_chunk(t, mesh, dtype):
    """`[1, heads, chunk_global, head_dim]` -> per-device `[1, heads/tp, chunk_local, head_dim]`.

    `dims=(2, 1)`: sequence over the SP **rows**, heads over the TP **columns** — the mapping the
    model's own chunk path produces, so the cache geometry under test is the deployment one.
    """
    rows, cols = tuple(mesh.shape)
    return ttnn.from_torch(
        t,
        device=mesh,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(2, 1)),
    )


def _gather_output(out, rows, cols, chunk_local, n_q_local, head_dim):
    """Per-device ring output -> `[1, n_q, chunk_global, head_dim]` in global position order."""
    per_device = ttnn.get_device_tensors(out)
    n_q = n_q_local * cols
    gathered = torch.zeros(1, n_q, chunk_local * rows, head_dim)
    for r in range(rows):
        for c in range(cols):
            block = ttnn.to_torch(per_device[r * cols + c]).float()
            assert tuple(block.shape) == (1, n_q_local, chunk_local, head_dim), (
                f"device ({r},{c}) returned {tuple(block.shape)}, expected "
                f"(1, {n_q_local}, {chunk_local}, {head_dim})"
            )
            gathered[0, c * n_q_local : (c + 1) * n_q_local, r * chunk_local : (r + 1) * chunk_local, :] = block[0]
    return gathered


def _populate(mesh, head_dim, *, n_kv, logical_n):
    """Build the global K/V, write them into a fresh cache chunk by chunk, return every host copy.

    Returns `(kv_cache, fp32, device_valued)`, each of the latter two a `(k, v)` pair:

    * `fp32` — the tensors as generated, which is what the **floor** is computed from;
    * `device_valued` — the same tensors quantised to `bfloat8_b`, i.e. **exactly the values the
      cache holds**, which is what the reference the device is scored against uses. The quantiser
      is applied in the tensors' own `[1, n_kv, seq, head_dim]` orientation, the one the device
      stores, because `bfloat8_b`'s shared exponent is per 16-element block of the **last** dim and
      quantising a permuted view would produce a different tensor (recipe §2.2.3a).
    """
    torch.manual_seed(0)
    k_global = torch.randn(1, n_kv, logical_n, head_dim)
    v_global = torch.randn(1, n_kv, logical_n, head_dim)

    kv_cache = allocate_kv_cache(
        mesh, num_layers=1, max_seq_len=CACHE_GLOBAL, num_users=1, head_dim=head_dim, cache_dtype=CACHE_DTYPE
    )
    assert logical_n % CHUNK_GLOBAL == 0, f"logical_n {logical_n} must be a whole number of {CHUNK_GLOBAL}-chunks"
    for chunk in range(logical_n // CHUNK_GLOBAL):
        lo, hi = chunk * CHUNK_GLOBAL, (chunk + 1) * CHUNK_GLOBAL
        tt_k = _shard_chunk(k_global[:, :, lo:hi, :], mesh, ACTIVATION_DTYPE)
        tt_v = _shard_chunk(v_global[:, :, lo:hi, :], mesh, ACTIVATION_DTYPE)
        write_kv_chunk(kv_cache, tt_k, tt_v, slot_idx=0, layer_idx=0, kv_actual=lo, sp_axis=0)
        tt_k.deallocate(True)
        tt_v.deallocate(True)
    ttnn.synchronize_device(mesh)
    return (
        kv_cache,
        (k_global, v_global),
        (quantize_like_device(k_global, CACHE_DTYPE), quantize_like_device(v_global, CACHE_DTYPE)),
    )


@requires_galaxy
@requires_ring_fabric
@torch.no_grad()
@pytest.mark.parametrize("device_params", [galaxy_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
def test_dense_sp_attention_vs_fp32_reference(mesh_device, reset_seeds):
    """**`G-SP-RING`.** The ring op alone, on the deployment `(4, 8)` mesh.

    * **Input distribution:** **standard-normal** Q/K/V, iid. Stated because it matters here more
      than anywhere else: recipe §2.3 measured that a *fused* kernel's error **ratio** is
      distribution-dependent even though the floor is not — the same single-card SDPA sits at 71x
      on iid standard-normal Q/K and 27-29x on real correlated post-RoPE activations. So this
      number is the iid one and is not comparable with an in-model measurement of the same op;
      `G-CHUNK-ATTN` is the in-model arm.
    * **Reference dtype policy:** Q rounded to **bf16** (what the projection hands the op), K/V
      rounded to **`bfloat8_b`** (what the cache holds), **all remaining math fp32** — the softmax,
      the two matmuls and the mask. No internal intermediate is quantised (§2.2's conservative
      reading), so this is a floor and not a flattered one.
    * **Computed noise floor:** the same reference re-run with Q/K/V *already* at their device
      values (which is what the reference above uses) scored against the fp32-input reference —
      i.e. the floor isolates the storage rounding from the kernel.
    * **Negative control:** two, in the tests below — `fp32_dest_acc_en=True` must be **refused**
      by the op (§1.4 counts a configuration that must refuse as a control), and a wrong
      `kv_actual_isl` must collapse the PCC.
    """
    hf = llama_config_dims()
    head_dim = derive_head_dim(hf)
    n_q, n_kv = hf["num_attention_heads"], hf["num_key_value_heads"]
    rows, cols = GALAXY_MESH_SHAPE
    sp, tp = rows, cols
    chunk_local = CHUNK_GLOBAL // sp
    n_q_local, n_kv_local = n_q // tp, n_kv // tp
    logical_n = KV_ACTUAL + CHUNK_GLOBAL
    scale = head_dim**-0.5
    gqa_group = n_q // n_kv

    assert n_kv_local == 1, f"the packed cache holds one KV head per chip; got {n_kv_local} at tp={tp}"
    assert chunk_local == 128, f"chunk_local {chunk_local} must equal the ring q_chunk_size (128)"

    mesh_config = MeshConfig(GALAXY_MESH_SHAPE, tp=tp)
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=prefill_topology())
    program_config = sp_ring_program_config(mesh_device)
    grid = mesh_device.compute_with_storage_grid_size()
    logger.info(
        f"[G-SP-RING] mesh={GALAXY_MESH_SHAPE} sp={sp} tp={tp} num_links={ccl.num_links} "
        f"topology={prefill_topology()} compute_grid=({grid.x},{grid.y}) "
        f"ccl_offset={ccl.ring_attention_ccl_core_grid_offset} "
        f"ring_sdpa_grid=({program_config.compute_with_storage_grid_size.x},"
        f"{program_config.compute_with_storage_grid_size.y}) "
        f"chunk_global={CHUNK_GLOBAL} chunk_local={chunk_local} kv_actual={KV_ACTUAL} "
        f"logical_n={logical_n} cache_global={CACHE_GLOBAL}"
    )

    kv_cache, (k_fp32, v_fp32), (k_dev, v_dev) = _populate(mesh_device, head_dim, n_kv=n_kv, logical_n=logical_n)
    torch.manual_seed(1)
    q_global = torch.randn(1, n_q, CHUNK_GLOBAL, head_dim)
    q_dev = quantize_like_device(q_global, ACTIVATION_DTYPE)
    tt_q = _shard_chunk(q_dev, mesh_device, ACTIVATION_DTYPE)

    out = dense_sp_attention(
        tt_q,
        kv_cache.k,
        kv_cache.v,
        None,
        None,
        kv_actual=KV_ACTUAL,
        logical_n=logical_n,
        n_kv=n_kv,
        cache_global=CACHE_GLOBAL,
        head_dim=head_dim,
        mesh_device=mesh_device,
        ccl_manager=ccl,
        program_config=program_config,
        compute_kernel_config=sp_ring_compute_kernel_config(mesh_device),
        scale=scale,
        cluster_axis=mesh_config.sp_axis,
        slot_idx=0,
        layer_idx=0,
        num_layers=1,
        write_chunk=False,
    )
    ttnn.synchronize_device(mesh_device)
    measured = _gather_output(out, rows, cols, chunk_local, n_q_local, head_dim)
    out.deallocate(True)
    tt_q.deallocate(True)
    kv_cache.k.deallocate(True)
    kv_cache.v.deallocate(True)

    # The reference, on the device's own stored values -> this is what `measured` is scored against.
    reference, _, _ = _torch_ring_reference(
        q_dev, k_dev, v_dev, kv_actual=KV_ACTUAL, logical_n=logical_n, gqa_group=gqa_group, scale=scale
    )
    # The floor: the same reference on the *un-rounded* fp32 inputs. The gap between the two is the
    # storage rounding alone, with no kernel in it.
    fp32_reference, _, _ = _torch_ring_reference(
        q_global, k_fp32, v_fp32, kv_actual=KV_ACTUAL, logical_n=logical_n, gqa_group=gqa_group, scale=scale
    )
    _, floor = comp_pcc(fp32_reference, reference, 0.0)
    _, pcc = comp_pcc(reference, measured, 0.0)
    ratio = err_ratio(float(pcc), float(floor))
    logger.info(
        f"[G-SP-RING] ring-joint SDPA alone: PCC = {float(pcc):.7f} (threshold "
        f"{SP_RING_PCC_THRESHOLD}); own noise floor = {float(floor):.7f} -> **{ratio:.2f}x**. "
        f"Recorded, not asserted: recipe section 2.3 measures that a fused kernel does not sit at "
        f"its floor (the single-card SDPA is 52.8-55.0x standalone at G-ATTN) and section 2.3.1 "
        f"that a ratio budget is not portable across dtypes. The recipe's figure for this op is 7.98x."
    )
    assert float(pcc) >= SP_RING_PCC_THRESHOLD, (
        f"the ring-joint SP attention scored PCC {float(pcc):.7f} < {SP_RING_PCC_THRESHOLD} against "
        f"an fp32 reference on its own stored values. Check the block-cyclic Q/cache position "
        f"mapping and kv_cache_batch_idx before suspecting the kernel."
    )


# =============================================================================================
# Control 1 — `fp32_dest_acc_en=True` must be REFUSED, structurally.
# =============================================================================================
@requires_galaxy
@requires_ring_fabric
@torch.no_grad()
@pytest.mark.parametrize("device_params", [galaxy_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
def test_ring_sdpa_refuses_fp32_dest_acc(mesh_device, reset_seeds):
    """The `fp32_dest_acc_en` A/B, with the `TT_FATAL` text recorded rather than paraphrased.

    This is the **one op in this model where `False` is mandatory** rather than the package default
    of `True` (recipe §2.4's closing line, `DEC-030`). It is not a regression of §2.4 but a
    structural constraint: `use_streaming_compute = !fp32_dest_acc_en`
    (`ring_joint_sdpa_program_factory.cpp:1304`) and `kv_actual_isl` — which every chunked call
    passes — requires the streaming path (`:1306-1308`).

    Recorded as an A/B rather than asserted away, and it doubles as `G-SP-RING`'s negative control
    (recipe §1.4: "an op or a configuration that must *refuse* counts as a control").
    """
    hf = llama_config_dims()
    head_dim = derive_head_dim(hf)
    n_q, n_kv = hf["num_attention_heads"], hf["num_key_value_heads"]
    logical_n = KV_ACTUAL + CHUNK_GLOBAL
    mesh_config = MeshConfig(GALAXY_MESH_SHAPE, tp=GALAXY_MESH_SHAPE[1])
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=prefill_topology())

    kv_cache, _, _ = _populate(mesh_device, head_dim, n_kv=n_kv, logical_n=logical_n)
    torch.manual_seed(1)
    tt_q = _shard_chunk(torch.randn(1, n_q, CHUNK_GLOBAL, head_dim), mesh_device, ACTIVATION_DTYPE)

    def _call(fp32_dest_acc_en):
        return dense_sp_attention(
            tt_q,
            kv_cache.k,
            kv_cache.v,
            None,
            None,
            kv_actual=KV_ACTUAL,
            logical_n=logical_n,
            n_kv=n_kv,
            cache_global=CACHE_GLOBAL,
            head_dim=head_dim,
            mesh_device=mesh_device,
            ccl_manager=ccl,
            program_config=sp_ring_program_config(mesh_device),
            compute_kernel_config=sp_ring_compute_kernel_config(mesh_device, fp32_dest_acc_en=fp32_dest_acc_en),
            scale=head_dim**-0.5,
            cluster_axis=mesh_config.sp_axis,
            num_layers=1,
        )

    refused = None
    try:
        _call(True).deallocate(True)
    except Exception as e:  # noqa: BLE001 — the message IS the measurement
        refused = str(e)
    logger.info(
        "[G-SP-RING] fp32_dest_acc_en=True: "
        + (f"REFUSED with:\n{refused}" if refused else "**ACCEPTED** — the recipe's claim does not hold on this build")
    )
    assert refused is not None, (
        "fp32_dest_acc_en=True was ACCEPTED by the ring op. The recipe states it is refused "
        "structurally (BRINGUP_RECIPE.md:1744-1751 via ring_joint_sdpa_program_factory.cpp:1304); "
        "if this build accepts it, dense_sp.py's default should be revisited with a measurement."
    )
    assert "streaming" in refused or "fp32_dest_acc_en" in refused, (
        f"fp32_dest_acc_en=True was refused, but not by the streaming-compute assert this gate "
        f"names. The message was: {refused}"
    )

    # And the same call with `False` must succeed, so the refusal above is about the flag and not
    # about the shapes.
    ok = _call(False)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"[G-SP-RING] fp32_dest_acc_en=False: accepted, output {tuple(ok.shape)}")
    ok.deallocate(True)
    tt_q.deallocate(True)
    kv_cache.k.deallocate(True)
    kv_cache.v.deallocate(True)


# =============================================================================================
# Control 2 — a wrong `kv_cache_batch_idx` must collapse the numbers.
#
# `dense_sp.py` fact 3: `kv_cache_batch_idx` must be `slot_idx * num_layers + layer_idx`, and
# passing the slot alone makes **every layer read layer 0's cache** — so layer 0 is correct by
# coincidence and layers 1+ read stale K/V. That is the exact shape of bug a single-layer or
# layer-0-only check cannot see, which is why it is this gate's numeric control.
#
# It replaces the control this file was first written with. A wrong `kv_actual_isl` turned out not
# to be a numeric control at all: the op **refuses** it —
#   TT_FATAL @ ring_joint_sdpa_device_operation.cpp:278: new_actual_isl <= chunk_capacity
#   "KV-pad-aware rotation expects current valid Q to fit in one fixed chunk.
#    Got new_actual_isl=1024, chunk capacity=512"
# — because `logical_n - kv_actual_isl` must fit one chunk. That refusal is recorded below as a
# second, structural control (§1.4 counts a configuration that must refuse as one), but a gate
# needs at least one control that the op *accepts* and gets wrong, and this is it (`DEC-082`).
# =============================================================================================
@requires_galaxy
@requires_ring_fabric
@torch.no_grad()
@pytest.mark.parametrize("device_params", [galaxy_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
def test_kv_cache_batch_idx_must_fold_in_the_layer(mesh_device, reset_seeds):
    """Two layers with different K/V; reading layer 1 with `batch_idx = slot` must collapse.

    The positive half is also load-bearing: it is the only place a **non-zero layer index** is
    proved to read its own cache slice, which `G-SP-RING`'s main test (one layer) cannot show.
    """
    hf = llama_config_dims()
    head_dim = derive_head_dim(hf)
    n_q, n_kv = hf["num_attention_heads"], hf["num_key_value_heads"]
    rows, cols = GALAXY_MESH_SHAPE
    chunk_local = CHUNK_GLOBAL // rows
    logical_n = KV_ACTUAL + CHUNK_GLOBAL
    scale = head_dim**-0.5
    gqa_group = n_q // n_kv
    num_layers = 2
    mesh_config = MeshConfig(GALAXY_MESH_SHAPE, tp=cols)
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=prefill_topology())

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=CACHE_GLOBAL,
        num_users=1,
        head_dim=head_dim,
        cache_dtype=CACHE_DTYPE,
    )
    per_layer = {}
    for layer_idx in range(num_layers):
        torch.manual_seed(10 + layer_idx)
        k_global = torch.randn(1, n_kv, logical_n, head_dim)
        v_global = torch.randn(1, n_kv, logical_n, head_dim)
        for chunk in range(logical_n // CHUNK_GLOBAL):
            lo, hi = chunk * CHUNK_GLOBAL, (chunk + 1) * CHUNK_GLOBAL
            tt_k = _shard_chunk(k_global[:, :, lo:hi, :], mesh_device, ACTIVATION_DTYPE)
            tt_v = _shard_chunk(v_global[:, :, lo:hi, :], mesh_device, ACTIVATION_DTYPE)
            write_kv_chunk(kv_cache, tt_k, tt_v, slot_idx=0, layer_idx=layer_idx, kv_actual=lo, sp_axis=0)
            tt_k.deallocate(True)
            tt_v.deallocate(True)
        per_layer[layer_idx] = (
            quantize_like_device(k_global, CACHE_DTYPE),
            quantize_like_device(v_global, CACHE_DTYPE),
        )
    ttnn.synchronize_device(mesh_device)

    torch.manual_seed(1)
    q_dev = quantize_like_device(torch.randn(1, n_q, CHUNK_GLOBAL, head_dim), ACTIVATION_DTYPE)
    tt_q = _shard_chunk(q_dev, mesh_device, ACTIVATION_DTYPE)

    def _run(layer_idx, layers_for_batch_idx):
        out = dense_sp_attention(
            tt_q,
            kv_cache.k,
            kv_cache.v,
            None,
            None,
            kv_actual=KV_ACTUAL,
            logical_n=logical_n,
            n_kv=n_kv,
            cache_global=CACHE_GLOBAL,
            head_dim=head_dim,
            mesh_device=mesh_device,
            ccl_manager=ccl,
            program_config=sp_ring_program_config(mesh_device),
            compute_kernel_config=sp_ring_compute_kernel_config(mesh_device),
            scale=scale,
            cluster_axis=mesh_config.sp_axis,
            slot_idx=0,
            layer_idx=layer_idx,
            num_layers=layers_for_batch_idx,
        )
        ttnn.synchronize_device(mesh_device)
        gathered = _gather_output(out, rows, cols, chunk_local, n_q // cols, head_dim)
        out.deallocate(True)
        return gathered

    # Correct: layer 1 with num_layers=2 -> kv_cache_batch_idx = 0*2 + 1 = 1.
    correct = _run(1, num_layers)
    # Control: layer 0 with num_layers=1 -> kv_cache_batch_idx = 0, i.e. "pass the slot alone".
    broken = _run(0, 1)
    tt_q.deallocate(True)
    kv_cache.k.deallocate(True)
    kv_cache.v.deallocate(True)

    k1, v1 = per_layer[1]
    reference, _, _ = _torch_ring_reference(
        q_dev, k1, v1, kv_actual=KV_ACTUAL, logical_n=logical_n, gqa_group=gqa_group, scale=scale
    )
    _, pcc_correct = comp_pcc(reference, correct, 0.0)
    _, pcc_broken = comp_pcc(reference, broken, 0.0)
    logger.info(
        f"[G-SP-RING] kv_cache_batch_idx: layer 1 read with batch_idx=slot*num_layers+layer "
        f"scores PCC = {float(pcc_correct):.7f}; the same read with batch_idx=slot (the template's "
        f"warned-about bug) scores PCC = {float(pcc_broken):.5f} against layer 1's reference. "
        f"Threshold {SP_RING_PCC_THRESHOLD}."
    )
    assert not math.isnan(float(pcc_broken)), "the control produced NaN; a NaN is not a collapse, it is a broken probe"
    assert float(pcc_correct) >= SP_RING_PCC_THRESHOLD, (
        f"reading layer 1's cache slice scored {float(pcc_correct):.7f} < {SP_RING_PCC_THRESHOLD}; "
        f"kv_cache_batch_idx = slot_idx*num_layers + layer_idx is not addressing the right slot"
    )
    assert float(pcc_broken) < SP_RING_PCC_THRESHOLD, (
        f"passing kv_cache_batch_idx = slot alone still scored {float(pcc_broken):.7f} — this gate "
        f"is not measuring which layer's cache the op reads, so it would miss the bug that makes "
        f"layer 0 correct by coincidence and every later layer stale"
    )


# =============================================================================================
# Control 3 — a wrong `kv_actual_isl` is REFUSED, and the refusal is the measurement.
# =============================================================================================
@requires_galaxy
@requires_ring_fabric
@torch.no_grad()
@pytest.mark.parametrize("device_params", [galaxy_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
def test_wrong_kv_actual_isl_is_refused(mesh_device, reset_seeds):
    """Lying about where this chunk starts is refused by the op, not silently mis-masked.

    Recorded rather than assumed: this is *not* what the test author expected — the plan was for
    the PCC to collapse. The op instead derives "current valid tokens" as
    `logical_n - kv_actual_isl` and requires that to fit one fixed chunk
    (`ring_joint_sdpa_device_operation.cpp:274`, reported as `:278`), so `kv_actual_isl=0` at `logical_n=1024` asks for
    1024 valid Q rows in a 512-row chunk and aborts. Good news for the deployment path — an
    off-by-`actual_start` chunk cannot silently produce a wrong answer through this op — and the
    reason `test_kv_cache_batch_idx_must_fold_in_the_layer` exists as the *numeric* control.
    """
    hf = llama_config_dims()
    head_dim = derive_head_dim(hf)
    n_q, n_kv = hf["num_attention_heads"], hf["num_key_value_heads"]
    logical_n = KV_ACTUAL + CHUNK_GLOBAL
    mesh_config = MeshConfig(GALAXY_MESH_SHAPE, tp=GALAXY_MESH_SHAPE[1])
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=prefill_topology())

    kv_cache, _, _ = _populate(mesh_device, head_dim, n_kv=n_kv, logical_n=logical_n)
    torch.manual_seed(1)
    tt_q = _shard_chunk(torch.randn(1, n_q, CHUNK_GLOBAL, head_dim), mesh_device, ACTIVATION_DTYPE)

    refused = None
    try:
        dense_sp_attention(
            tt_q,
            kv_cache.k,
            kv_cache.v,
            None,
            None,
            kv_actual=0,  # the lie: the chunk really starts at KV_ACTUAL
            logical_n=logical_n,
            n_kv=n_kv,
            cache_global=CACHE_GLOBAL,
            head_dim=head_dim,
            mesh_device=mesh_device,
            ccl_manager=ccl,
            program_config=sp_ring_program_config(mesh_device),
            compute_kernel_config=sp_ring_compute_kernel_config(mesh_device),
            scale=head_dim**-0.5,
            cluster_axis=mesh_config.sp_axis,
            num_layers=1,
        ).deallocate(True)
    except Exception as e:  # noqa: BLE001 — the message IS the measurement
        refused = str(e)
    tt_q.deallocate(True)
    kv_cache.k.deallocate(True)
    kv_cache.v.deallocate(True)

    logger.info(
        f"[G-SP-RING] control kv_actual_isl=0 (true value {KV_ACTUAL}): "
        + (f"REFUSED with:\n{refused}" if refused else "**ACCEPTED** — see the assertion below")
    )
    assert refused is not None, (
        "the op accepted kv_actual_isl=0 for a chunk that starts at 512. Then an "
        "off-by-actual_start chunk CAN be silently mis-masked through this path, and "
        "tt_prefill_runtime.py's own tile-alignment and range checks are the only thing standing "
        "between the engine and a wrong answer. Re-score this control numerically."
    )
    assert "new_actual_isl" in refused or "chunk capacity" in refused, (
        f"kv_actual_isl=0 was refused, but not by the KV-pad-aware rotation check this control "
        f"names. The message was: {refused}"
    )
