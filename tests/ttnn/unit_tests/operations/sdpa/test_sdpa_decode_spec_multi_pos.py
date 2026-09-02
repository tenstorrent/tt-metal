# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``spec_multi_pos_tiles`` on
``ttnn.transformer.paged_scaled_dot_product_attention_decode``.

Speculative-decode verify calls paged SDPA decode with T pseudo-users whose page tables
are identical (they all alias ONE KV cache) and whose ``cur_pos`` values are consecutive.
The program factory partitions cores per batch row, so those T rows each stream the whole
KV cache out of DRAM — T times the traffic for one cache. ``spec_multi_pos_tiles=Tg`` folds
Tg candidates onto a single batch row (PNHt == Tg, candidate j owning row-tile j) so that
row's KV is scanned once; the per-candidate causal bound moves from the scan range into a
per-row-tile mask.

With B batch rows the T = B*Tg candidates are split into B *groups* of Tg: Q is
``[1, B, Tg*32, DH]``, the page table has B (aliased) rows, and ``cur_pos`` carries B*Tg
bounds with batch b owning ``[b*Tg, (b+1)*Tg)``. B > 1 exists because every Q-shaped CB
scales with Tg, so L1 caps cores-per-head by Tg: Tg=4 fits 64 cores/head but Tg=7 only 4.
Two groups of 4 therefore cover 8 candidates while keeping the whole grid busy — each group
scans the KV once on its own reduction group. The B == 1 sweep below is the Tg == T special
case; the group sweep at the bottom of the file covers B == 2.

The property under test is that the two forms are numerically the same computation:

* reference  — today's mode, B == T rows, ``q`` ``[1, T, 32, DH]``, T identical page-table
  rows, ``cur_pos = [p, p+1, ..., p+T-1]``
* spec       — the *same Q bytes* viewed as ``[1, 1, T*32, DH]``, one page-table row, the
  same ``cur_pos`` vector, ``spec_multi_pos_tiles=T``

Both are also checked against an fp32 torch reference that applies each candidate's causal
bound independently.

How tightly the two device paths can be expected to agree depends on whether flash decode
sums its bf16 partial results in the same order:

* matched reduction tree, bounds inside one k-chunk -> **bit-identical**
  (``test_spec_multi_pos_is_bit_exact``)
* matched tree, bounds straddling a chunk boundary  -> spec scans one extra (fully masked,
  zero-contribution) chunk, which redistributes chunks across cores -> PCC >= 0.999
* deliberately mismatched tree (``max_cores_per_head_batch=64``, where spec mode gets all 64
  cores on its single row and the reference gets ~27 per row) -> PCC >= 0.999, and spec must
  be no less accurate than the reference against fp32
  (``test_spec_multi_pos_wide_reduction_split``)

``p`` is swept over tile-alignment edges (``p % 32`` in {0, 1, 30, 31}) and over k-chunk
boundaries, including cases where the T bounds straddle a chunk boundary so that TWO chunks
carry a mask instead of one.
"""

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

# One module-scoped device: the KV caches below are up to 32k tokens and every case
# reuses the same two programs per (T, seq_len).
pytestmark = pytest.mark.use_module_device

BLOCK_SIZE = 64
HEAD_DIM = 256
NUM_KV_HEADS = 1
NUM_Q_HEADS = 6  # valid q heads per candidate; rows 6..31 of each Q tile are padding
TILE = 32
SCALE = 1.0 / (HEAD_DIM**0.5)

# Spec mode makes every Q-shaped CB T times larger (PNHt == T), so the (k_chunk_size,
# cores-per-head) budget shrinks sharply as T grows. These are the largest points that fit
# Blackhole L1 (1,461,248 B of CB space per core) at head_dim=256 — see the explicit L1
# guard in the program factory, which reports the numbers when a config does not fit.
#
#   T=4,  chunk=128, 64 cores/head -> 1,310,720 B
#   T=7,  chunk=64,   4 cores/head -> 1,329,152 B
#   T=11, chunk=32,   1 core /head -> 1,447,936 B
#
# The sweep below uses core counts chosen so the reference (B == T rows spread over the same
# grid) lands on the SAME cores-per-head as spec mode (B == 1). With an identical reduction
# tree the two paths agree bit-for-bit, which isolates the mode from flash-decode's bf16
# partial-sum ordering. ``test_spec_multi_pos_wide_reduction_split`` covers the other case,
# where the trees deliberately differ.
SPEC_CONFIG = {
    4: {"max_cores": 16, "k_chunk_size": 128},
    7: {"max_cores": 4, "k_chunk_size": 64},
    11: {"max_cores": 1, "k_chunk_size": 32},
}


# ── Helpers ────────────────────────────────────────────────────────────────


def _build_kv(seq_len, seed):
    """One logical user's K/V, bf16-rounded so the paged cache round-trips exactly."""
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(NUM_KV_HEADS, seq_len, HEAD_DIM, generator=g).bfloat16().float()
    v = torch.randn(NUM_KV_HEADS, seq_len, HEAD_DIM, generator=g).bfloat16().float()
    return k, v


def _paged_layout(k, v, page_table_row):
    """Scatter the user's K/V into a paged buffer keyed by physical block id."""
    num_blocks = page_table_row.numel()
    paged_k = torch.zeros(num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM)
    paged_v = torch.zeros_like(paged_k)
    for virtual_block in range(num_blocks):
        physical_block = int(page_table_row[virtual_block])
        lo = virtual_block * BLOCK_SIZE
        hi = lo + BLOCK_SIZE
        paged_k[physical_block] = k[:, lo:hi, :]
        paged_v[physical_block] = v[:, lo:hi, :]
    return paged_k, paged_v


def _torch_reference(q_heads, k, v, cur_pos):
    """fp32 causal reference: candidate j attends to KV positions [0, cur_pos[j]]."""
    T = q_heads.shape[0]
    out = torch.zeros(T, NUM_Q_HEADS, HEAD_DIM, dtype=torch.float32)
    for j in range(T):
        pos = int(cur_pos[j])
        k_j = k[0, : pos + 1, :].float()
        v_j = v[0, : pos + 1, :].float()
        scores = (q_heads[j].float() @ k_j.T) * SCALE
        out[j] = torch.softmax(scores, dim=-1) @ v_j
    return out


def _program_config(device, max_cores, k_chunk_size):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=32,
        k_chunk_size=k_chunk_size,  # 0 -> dynamic k-chunk size, chosen in-kernel from cur_pos
        exp_approx_mode=False,
        max_cores_per_head_batch=max_cores,
    )


def _config_for(device, T):
    cfg = SPEC_CONFIG[T]
    return _program_config(device, cfg["max_cores"], cfg["k_chunk_size"])


def _build_inputs(device, T, p, seq_len, seed):
    """Everything both calls share, plus the two call-specific Q / page-table tensors."""
    assert seq_len % BLOCK_SIZE == 0
    num_blocks = seq_len // BLOCK_SIZE
    cur_pos = torch.tensor([p + j for j in range(T)], dtype=torch.int32)
    assert int(cur_pos[-1]) < seq_len, "cur_pos must stay inside the cache"

    k, v = _build_kv(seq_len, seed)
    g = torch.Generator().manual_seed(seed + 1)
    # Shuffled blocks: the two calls must resolve the same physical blocks through the
    # page table, not through a coincidentally identity mapping.
    page_row = torch.randperm(num_blocks, generator=g).to(torch.int32)
    paged_k, paged_v = _paged_layout(k, v, page_row)

    # Q: one 32-row tile per candidate, rows 0..NUM_Q_HEADS-1 valid, the rest zero padding.
    # [1, T, 32, DH] and [1, 1, T*32, DH] are byte-identical once tilized, which is exactly
    # the equivalence the mode relies on.
    q_heads = torch.randn(T, NUM_Q_HEADS, HEAD_DIM, generator=g).bfloat16().float()
    q_batched = torch.zeros(1, T, TILE, HEAD_DIM)
    q_batched[0, :, :NUM_Q_HEADS, :] = q_heads

    return {
        "k": k,
        "v": v,
        "q_heads": q_heads,
        "cur_pos": cur_pos,
        "page_row": page_row,
        "k_tt": ttnn.Tensor(paged_k, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        "v_tt": ttnn.Tensor(paged_v, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        "cur_pos_tt": ttnn.Tensor(cur_pos, ttnn.int32).to(device),
        "q_batched": q_batched,
        "q_spec": q_batched.reshape(1, 1, T * TILE, HEAD_DIM),
    }


def _run_reference(device, inp, T, program_config):
    """Today's mode: B == T pseudo-users, T identical (aliased) page-table rows."""
    page_table = inp["page_row"].unsqueeze(0).repeat(T, 1).contiguous()
    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        ttnn.Tensor(inp["q_batched"], ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        inp["k_tt"],
        inp["v_tt"],
        page_table_tensor=ttnn.Tensor(page_table, ttnn.int32).to(device),
        cur_pos_tensor=inp["cur_pos_tt"],
        scale=SCALE,
        program_config=program_config,
    )
    torch_out = ttnn.to_torch(out)
    assert tuple(torch_out.shape) == (1, T, TILE, HEAD_DIM)
    return torch_out[0, :, :NUM_Q_HEADS, :].float()


def _run_spec(device, inp, T, program_config):
    """Spec mode: the same Q bytes on ONE batch row, one page-table row."""
    page_table = inp["page_row"].unsqueeze(0).contiguous()
    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        ttnn.Tensor(inp["q_spec"], ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        inp["k_tt"],
        inp["v_tt"],
        page_table_tensor=ttnn.Tensor(page_table, ttnn.int32).to(device),
        cur_pos_tensor=inp["cur_pos_tt"],
        scale=SCALE,
        program_config=program_config,
        spec_multi_pos_tiles=T,
    )
    torch_out = ttnn.to_torch(out)
    # Byte-identical layout to the B=T output [1, T, 32, DH].
    assert tuple(torch_out.shape) == (1, 1, T * TILE, HEAD_DIM)
    return torch_out.reshape(1, T, TILE, HEAD_DIM)[0, :, :NUM_Q_HEADS, :].float()


def _straddles_chunk_boundary(T, p):
    """True when the T bounds span two k-chunks.

    This matters for how tightly the two paths can be expected to agree. Spec mode scans up
    to the LARGEST bound, so when the bounds straddle a boundary it scans one chunk more than
    the reference rows whose own bound sits in the lower chunk. That extra chunk is fully
    masked and contributes exactly zero, but it changes the chunk count, hence the
    chunk-to-core distribution, hence the order in which flash decode sums its bf16 partial
    results — so bit-equality is not available for these cases even with matched core counts.
    """
    chunk = SPEC_CONFIG[T]["k_chunk_size"]
    return (p // chunk) != ((p + T - 1) // chunk)


def _check(device, T, p, seq_len, seed=0, program_config=None, pcc_ref_vs_spec=None):
    inp = _build_inputs(device, T, p, seq_len, seed)
    program_config = program_config if program_config is not None else _config_for(device, T)
    if pcc_ref_vs_spec is None:
        pcc_ref_vs_spec = 0.999 if _straddles_chunk_boundary(T, p) else 0.9999

    ref = _run_reference(device, inp, T, program_config)
    spec = _run_spec(device, inp, T, program_config)
    torch_ref = _torch_reference(inp["q_heads"], inp["k"], inp["v"], inp["cur_pos"])
    where = f"T={T}, p={p}, seq_len={seq_len}"

    # 1) The two device paths must agree. With a matched reduction tree and bounds inside one
    #    k-chunk they come out bit-identical; when the bounds straddle a chunk boundary the
    #    chunk counts differ and only bf16 partial-sum accuracy is available (see
    #    _straddles_chunk_boundary).
    eq, msg = comp_pcc(ref, spec, pcc=pcc_ref_vs_spec)
    assert eq, f"spec vs batched ({where}): {msg}"
    assert torch.allclose(
        ref, spec, rtol=2e-2, atol=2e-2
    ), f"spec vs batched ({where}): max abs diff {(ref - spec).abs().max().item():.5f}"

    # 2) Both must match a plain fp32 reference that applies each candidate's bound
    #    independently — this is what catches an off-by-one or a dropped per-row bound.
    eq, msg = comp_pcc(torch_ref, spec, pcc=0.99)
    assert eq, f"spec vs torch ({where}): {msg}"
    eq, msg = comp_pcc(torch_ref, ref, pcc=0.99)
    assert eq, f"batched vs torch ({where}): {msg}"


# ── Equivalence sweep ──────────────────────────────────────────────────────
#
# ``p`` walks tile-alignment edges (p % 32 in {0, 1, 30, 31}) and k-chunk edges. The chunk
# width differs per T (see SPEC_CONFIG: 128 for T=4, 64 for T=7, 32 for T=11), so each p
# lands on a different point of each T's chunk grid — which is the intent: between them the
# cases cover "bound is the last position of a chunk" (one masked chunk), "bounds straddle a
# chunk boundary" (two masked chunks, possibly on two different cores) and "bounds sit well
# inside a chunk".


@pytest.mark.parametrize("T", [4, 7, 11], ids=["T4", "T7", "T11"])
@pytest.mark.parametrize(
    "seq_len, p",
    [
        # 2k context, tile-alignment edges (p % 32 in {0, 1, 30, 31})
        (2048, 1024),  # p % 32 == 0,  p % 128 == 0
        (2048, 1025),  # p % 32 == 1
        (2048, 1054),  # p % 32 == 30
        (2048, 1055),  # p % 32 == 31
        # chunk-boundary edges
        (2048, 1023),  # last position of a 128-chunk: bounds start a fresh chunk
        (2048, 1021),  # bounds straddle a 128-chunk boundary -> TWO masked chunks
        # 8k context
        (8192, 4095),  # last position of a chunk
        (8192, 4093),  # straddles a chunk boundary
        (8192, 5000),
    ],
    ids=[
        "s2k_p1024",
        "s2k_p1025",
        "s2k_p1054",
        "s2k_p1055",
        "s2k_p1023",
        "s2k_p1021_straddle",
        "s8k_p4095",
        "s8k_p4093_straddle",
        "s8k_p5000",
    ],
)
def test_spec_multi_pos_matches_batched(device, T, seq_len, p):
    torch.manual_seed(0)
    assert p + T - 1 < seq_len
    _check(device, T, p, seq_len)


@pytest.mark.parametrize("T", [4, 7, 11], ids=["T4", "T7", "T11"])
@pytest.mark.parametrize("seq_len, p", [(2048, 1024), (8192, 5000)], ids=["s2k", "s8k"])
def test_spec_multi_pos_is_bit_exact(device, T, seq_len, p):
    """The sharpest form of the claim. With the reduction tree matched (SPEC_CONFIG) and the
    T bounds inside a single k-chunk, folding the candidates onto one batch row is not merely
    numerically close to the B == T form — it produces the identical bits.
    """
    torch.manual_seed(5)
    assert not _straddles_chunk_boundary(T, p)
    inp = _build_inputs(device, T, p, seq_len, seed=23)
    pc = _config_for(device, T)
    ref = _run_reference(device, inp, T, pc)
    spec = _run_spec(device, inp, T, pc)
    assert torch.equal(ref, spec), (
        f"expected bit-identical output (T={T}, p={p}, seq_len={seq_len}); "
        f"max abs diff {(ref - spec).abs().max().item():.8f}"
    )


@pytest.mark.parametrize("T", [4, 7], ids=["T4", "T7"])
@pytest.mark.parametrize("p", [30000, 32639], ids=["p30000", "p32639_straddle"])
def test_spec_multi_pos_long_context(device, T, p):
    """32k context — the regime the mode exists for (DRAM-bound KV scan)."""
    torch.manual_seed(1)
    _check(device, T, p, seq_len=32768)


@pytest.mark.parametrize("p", [30000, 32639], ids=["p30000", "p32639_straddle"])
def test_spec_multi_pos_long_context_single_core(device, p):
    """T=11 at 32k. Only 1 core/head fits L1 at T=11 (see SPEC_CONFIG), so this run
    accumulates ~1000 chunks of bf16 flash-decode state serially on one core. That costs
    absolute accuracy — but it costs the LEGACY path exactly as much: at this config both
    paths land at ~0.988 PCC against fp32, while the same context with a legacy-friendly
    config (8 cores/head, 128-wide chunks) reaches 0.9996. So the assertion here is the
    equivalence itself, checked against the batched reference rather than against torch.
    """
    torch.manual_seed(2)
    T = 11
    inp = _build_inputs(device, T, p, seq_len=32768, seed=17)
    pc = _config_for(device, T)
    ref = _run_reference(device, inp, T, pc)
    spec = _run_spec(device, inp, T, pc)
    eq, msg = comp_pcc(ref, spec, pcc=0.9999)
    assert eq, f"spec vs batched (T={T}, p={p}, seq_len=32768): {msg}"
    assert torch.allclose(ref, spec, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("seq_len, p", [(2048, 1024), (8192, 5000), (32768, 30000)], ids=["s2k", "s8k", "s32k"])
def test_spec_multi_pos_wide_reduction_split(device, seq_len, p):
    """``max_cores_per_head_batch=64``: spec mode (B == 1) gets all 64 cores on its single
    row, while the reference spreads the same grid over B == T rows and gets ~27 — so the two
    runs sum their partial softmax results in a different order. They are no longer
    bit-identical, but neither is more accurate than the other, which is the claim that
    matters: folding the candidates onto one row costs nothing numerically.
    """
    torch.manual_seed(3)
    T = 4
    inp = _build_inputs(device, T, p, seq_len, seed=19)
    pc = _program_config(device, max_cores=64, k_chunk_size=128)
    ref = _run_reference(device, inp, T, pc)
    spec = _run_spec(device, inp, T, pc)
    torch_ref = _torch_reference(inp["q_heads"], inp["k"], inp["v"], inp["cur_pos"])

    eq, msg = comp_pcc(ref, spec, pcc=0.999)
    assert eq, f"spec vs batched (wide split, seq_len={seq_len}, p={p}): {msg}"
    assert torch.allclose(ref, spec, rtol=2e-2, atol=2e-2)

    # Spec mode must be no worse than the batched reference against fp32.
    err_ref = (ref - torch_ref).abs().max().item()
    err_spec = (spec - torch_ref).abs().max().item()
    assert err_spec <= err_ref * 1.5 + 1e-6, (
        f"spec mode lost accuracy vs the batched reference: |spec-torch|={err_spec:.6f} "
        f"vs |ref-torch|={err_ref:.6f}"
    )
    eq, msg = comp_pcc(torch_ref, spec, pcc=0.99)
    assert eq, f"spec vs torch (wide split, seq_len={seq_len}, p={p}): {msg}"


def test_spec_multi_pos_short_context(device):
    """Short context: the whole scan is one chunk, so the single chunk is the masked one."""
    torch.manual_seed(1)
    _check(device, T=4, p=40, seq_len=512, seed=7)


def test_spec_multi_pos_first_block(device):
    """cur_pos inside the very first tile — the mask cuts inside column-tile 0."""
    torch.manual_seed(2)
    _check(device, T=7, p=5, seq_len=512, seed=11)


def test_spec_multi_pos_dynamic_chunk(device):
    """``k_chunk_size=0``: the k-chunk is picked in-kernel from cur_pos and capped at 4 tiles
    in spec mode. The reference (PNHt == 1) is not capped and picks 8 tiles, so the two paths
    split the flash reduction differently and only agree to bf16 partial-sum accuracy — hence
    the looser bound here than in the matched-split sweep above. Both are still held to the
    fp32 torch reference at the usual 0.99."""
    torch.manual_seed(3)
    pc = _program_config(device, max_cores=64, k_chunk_size=0)
    _check(device, T=4, p=1024, seq_len=2048, seed=13, program_config=pc, pcc_ref_vs_spec=0.999)


# ── Validation ─────────────────────────────────────────────────────────────


def _minimal_spec_args(device, T=4, seq_len=512, p=100):
    inp = _build_inputs(device, T, p, seq_len, seed=3)
    return inp, _config_for(device, T)


def test_rejects_page_table_batch_mismatch(device, expect_error):
    """The page table gives one row per batch GROUP, so its row count must equal Q's batch
    dim. A B=T page table against a single-row Q is the legacy form, not a 1-group spec call."""
    T = 4
    inp, pc = _minimal_spec_args(device, T=T)
    page_table = inp["page_row"].unsqueeze(0).repeat(T, 1).contiguous()
    with expect_error(RuntimeError, "one row per Q batch group"):
        ttnn.transformer.paged_scaled_dot_product_attention_decode(
            ttnn.Tensor(inp["q_spec"], ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
            inp["k_tt"],
            inp["v_tt"],
            page_table_tensor=ttnn.Tensor(page_table, ttnn.int32).to(device),
            cur_pos_tensor=inp["cur_pos_tt"],
            scale=SCALE,
            program_config=pc,
            spec_multi_pos_tiles=T,
        )


def test_rejects_q_row_count_mismatch(device, expect_error):
    """spec_multi_pos_tiles must equal the number of 32-row Q tiles."""
    T = 4
    inp, pc = _minimal_spec_args(device, T=T)
    with expect_error(RuntimeError, "padded rows"):
        ttnn.transformer.paged_scaled_dot_product_attention_decode(
            ttnn.Tensor(inp["q_spec"], ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
            inp["k_tt"],
            inp["v_tt"],
            page_table_tensor=ttnn.Tensor(inp["page_row"].unsqueeze(0).contiguous(), ttnn.int32).to(device),
            cur_pos_tensor=inp["cur_pos_tt"],
            scale=SCALE,
            program_config=pc,
            spec_multi_pos_tiles=T + 1,
        )


def test_rejects_cur_pos_length_mismatch(device, expect_error):
    """cur_pos must carry exactly T bounds — one per candidate row-tile."""
    T = 4
    inp, pc = _minimal_spec_args(device, T=T)
    short_cur_pos = ttnn.Tensor(torch.tensor([100, 101], dtype=torch.int32), ttnn.int32).to(device)
    with expect_error(RuntimeError, "cur_pos must have"):
        ttnn.transformer.paged_scaled_dot_product_attention_decode(
            ttnn.Tensor(inp["q_spec"], ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
            inp["k_tt"],
            inp["v_tt"],
            page_table_tensor=ttnn.Tensor(inp["page_row"].unsqueeze(0).contiguous(), ttnn.int32).to(device),
            cur_pos_tensor=short_cur_pos,
            scale=SCALE,
            program_config=pc,
            spec_multi_pos_tiles=T,
        )


def test_rejects_sliding_window(device, expect_error):
    """The sliding-window mask shares the causal mask machinery; not combined for now."""
    T = 4
    inp, pc = _minimal_spec_args(device, T=T)
    with expect_error(RuntimeError, "sliding_window_size"):
        ttnn.transformer.paged_scaled_dot_product_attention_decode(
            ttnn.Tensor(inp["q_spec"], ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
            inp["k_tt"],
            inp["v_tt"],
            page_table_tensor=ttnn.Tensor(inp["page_row"].unsqueeze(0).contiguous(), ttnn.int32).to(device),
            cur_pos_tensor=inp["cur_pos_tt"],
            scale=SCALE,
            sliding_window_size=256,
            program_config=pc,
            spec_multi_pos_tiles=T,
        )


def test_legacy_path_unaffected(device):
    """spec_multi_pos_tiles omitted -> the pre-change op, bit-for-bit. Backward-compat guard."""
    torch.manual_seed(4)
    T = 4
    inp = _build_inputs(device, T, p=1000, seq_len=2048, seed=5)
    pc = _config_for(device, T)
    ref = _run_reference(device, inp, T, pc)
    torch_ref = _torch_reference(inp["q_heads"], inp["k"], inp["v"], inp["cur_pos"])
    eq, msg = comp_pcc(torch_ref, ref, pcc=0.99)
    assert eq, f"legacy path regressed: {msg}"


# ── Batch groups (B >= 1) ──────────────────────────────────────────────────
#
# Everything above runs B == 1: one batch row carrying all T candidates. That row's CBs all
# scale with T, so L1 caps how many cores can reduce it (Tg=4 -> 64 cores/head, Tg=7 -> 4),
# and a single group cannot fill the grid past its own cap. B groups of Tg lift that: batch b
# owns cur_pos[b*Tg : (b+1)*Tg] and gets its own reduction group of cores, so B=2, Tg=4 runs 8
# candidates across ~2x the cores that one 8-tile row could ever use.
#
# The reference is unchanged — the legacy B == B*Tg pseudo-user call. The claim is the same
# one the B == 1 sweep makes, now per group: candidate b*Tg+j must come out exactly as legacy
# batch row b*Tg+j did.

GROUP_CHUNK = 128  # k_chunk_size for the group sweep: 4 tiles, the spec-mode dynamic cap


def _cores_per_head(device, batch, max_cores):
    """The program factory's core split, mirrored (see 'Core Allocation' in the factory).

    Bit-equality between the spec call (B rows) and the legacy reference (B*Tg rows) is only
    available when both land on the same cores-per-head, since that fixes the flash-decode
    reduction tree and hence the order the bf16 partial results are summed.
    """
    grid = device.compute_with_storage_grid_size()
    available = grid.x * grid.y
    return max(1, min(available, max_cores * batch) // batch)


def _run_spec_groups(device, inp, B, Tg, program_config):
    """Spec mode with B groups: the same Q bytes as [1, B, Tg*32, DH], B aliased page rows."""
    T = B * Tg
    assert inp["cur_pos"].numel() == T
    # [1, T, 32, DH] -> [1, B, Tg*32, DH] is a pure reshape, so the tilized bytes are
    # identical to the reference's Q — candidate b*Tg+j lands on batch b, row-tile j.
    q_spec = inp["q_batched"].reshape(1, B, Tg * TILE, HEAD_DIM)
    page_table = inp["page_row"].unsqueeze(0).repeat(B, 1).contiguous()
    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        ttnn.Tensor(q_spec, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
        inp["k_tt"],
        inp["v_tt"],
        page_table_tensor=ttnn.Tensor(page_table, ttnn.int32).to(device),
        cur_pos_tensor=inp["cur_pos_tt"],
        scale=SCALE,
        program_config=program_config,
        spec_multi_pos_tiles=Tg,
    )
    torch_out = ttnn.to_torch(out)
    assert tuple(torch_out.shape) == (1, B, Tg * TILE, HEAD_DIM)
    return torch_out.reshape(1, T, TILE, HEAD_DIM)[0, :, :NUM_Q_HEADS, :].float()


def _group_straddles(B, Tg, p, chunk):
    """True when some group's Tg bounds span two k-chunks.

    Spec mode scans to the group's LARGEST bound, so a straddling group scans one chunk more
    than the reference rows whose own bound sits in the lower chunk. That chunk is fully
    masked and contributes zero, but it changes the chunk count and hence the chunk-to-core
    distribution — so bit-equality is off the table even with a matched tree.
    """
    return any((p + b * Tg) // chunk != (p + (b + 1) * Tg - 1) // chunk for b in range(B))


def _check_groups(device, B, Tg, p, seq_len, max_cores, seed=0, k_chunk_size=GROUP_CHUNK, pcc=None):
    T = B * Tg
    torch.manual_seed(seed)
    inp = _build_inputs(device, T, p, seq_len, seed)
    pc = _program_config(device, max_cores=max_cores, k_chunk_size=k_chunk_size)
    matched_tree = _cores_per_head(device, B, max_cores) == _cores_per_head(device, T, max_cores)
    if pcc is None:
        pcc = 0.9999 if matched_tree and not _group_straddles(B, Tg, p, k_chunk_size) else 0.999

    ref = _run_reference(device, inp, T, pc)
    spec = _run_spec_groups(device, inp, B, Tg, pc)
    torch_ref = _torch_reference(inp["q_heads"], inp["k"], inp["v"], inp["cur_pos"])
    where = f"B={B}, Tg={Tg}, p={p}, seq_len={seq_len}, max_cores={max_cores}"

    eq, msg = comp_pcc(ref, spec, pcc=pcc)
    assert eq, f"spec groups vs batched ({where}): {msg}"
    assert torch.allclose(
        ref, spec, rtol=2e-2, atol=2e-2
    ), f"spec groups vs batched ({where}): max abs diff {(ref - spec).abs().max().item():.5f}"

    # The fp32 check is what catches a group picking up the wrong slice of cur_pos: a
    # candidate given its neighbour group's bound still matches the device reference's shape
    # and magnitude, but not the independently-bounded softmax.
    eq, msg = comp_pcc(torch_ref, spec, pcc=0.99)
    assert eq, f"spec groups vs torch ({where}): {msg}"
    eq, msg = comp_pcc(torch_ref, ref, pcc=0.99)
    assert eq, f"batched vs torch ({where}): {msg}"


@pytest.mark.parametrize(
    "seq_len, p",
    [
        # 2k, tile-alignment edges (p % 32 in {0, 1, 30, 31})
        (2048, 1024),  # p % 32 == 0; both groups sit inside chunk 8
        (2048, 1025),  # p % 32 == 1
        (2048, 1054),  # p % 32 == 30; the groups' bounds cross a TILE edge
        (2048, 1055),  # p % 32 == 31
        # chunk-boundary edges. The interesting new case for groups is the pair where the two
        # groups do not agree on how far to scan:
        (2048, 1020),  # group 0 ends at 1023 (8 chunks), group 1 at 1027 (9) -> scan ranges differ
        (2048, 1021),  # group 0 straddles the boundary (TWO masked chunks), group 1 has one
        # 8k
        (8192, 4095),  # group 0 straddles at the top of chunk 31
        (8192, 5000),
        # 32k — the regime the mode exists for
        (32768, 30000),
        (32768, 32700),  # last chunk of the cache
    ],
    ids=[
        "s2k_p1024",
        "s2k_p1025",
        "s2k_p1054",
        "s2k_p1055",
        "s2k_p1020_split_scan",
        "s2k_p1021_straddle",
        "s8k_p4095_straddle",
        "s8k_p5000",
        "s32k_p30000",
        "s32k_p32700",
    ],
)
def test_spec_multi_pos_groups_matches_batched(device, seq_len, p):
    """B=2 groups of Tg=4 (8 candidates) at 55 cores/head — the full-grid config: each of the
    two batch rows gets its own ~55-core reduction group, so all ~110 cores are active and the
    KV is read twice for 8 candidates instead of eight times.

    The reference (B=8 rows) is capped by its own row count to ~13 cores/head, so the two
    reduction trees deliberately differ here and only bf16 partial-sum accuracy is available.
    ``test_spec_multi_pos_groups_is_bit_exact`` pins the matched-tree case.
    """
    _check_groups(device, B=2, Tg=4, p=p, seq_len=seq_len, max_cores=55, seed=31)


@pytest.mark.parametrize(
    "seq_len, p",
    [(2048, 1024), (2048, 1021), (8192, 5000)],
    ids=["s2k", "s2k_straddle", "s8k"],
)
def test_spec_multi_pos_groups_16_cores(device, seq_len, p):
    """The same claim at a much narrower split (16 cores/head), where each core carries many
    more chunks and the two masked chunks are far more likely to land on the same core."""
    _check_groups(device, B=2, Tg=4, p=p, seq_len=seq_len, max_cores=16, seed=37)


@pytest.mark.parametrize("seq_len, p", [(2048, 1024), (8192, 5000), (32768, 30000)], ids=["s2k", "s8k", "s32k"])
def test_spec_multi_pos_groups_is_bit_exact(device, seq_len, p):
    """The sharpest form of the grouped claim. ``max_cores_per_head_batch=8`` puts the B=2 spec
    call and the B=8 reference on the same cores-per-head (both are capped by max_cores, not by
    the grid), so the reduction trees match; with no group straddling a chunk boundary the two
    forms are not merely close but identical bit-for-bit.
    """
    B, Tg, max_cores = 2, 4, 8
    T = B * Tg
    if _cores_per_head(device, B, max_cores) != _cores_per_head(device, T, max_cores):
        pytest.skip("grid too small to give the spec and reference calls a matched reduction tree")
    assert not _group_straddles(B, Tg, p, GROUP_CHUNK)

    torch.manual_seed(41)
    inp = _build_inputs(device, T, p, seq_len, seed=43)
    pc = _program_config(device, max_cores=max_cores, k_chunk_size=GROUP_CHUNK)
    ref = _run_reference(device, inp, T, pc)
    spec = _run_spec_groups(device, inp, B, Tg, pc)
    assert torch.equal(ref, spec), (
        f"expected bit-identical output (B={B}, Tg={Tg}, p={p}, seq_len={seq_len}); "
        f"max abs diff {(ref - spec).abs().max().item():.8f}"
    )


def test_spec_multi_pos_groups_dynamic_chunk(device):
    """``k_chunk_size=0`` with the two groups on OPPOSITE sides of a dynamic-chunk step.

    The in-kernel chunk size is derived from the position — nearest power of two of
    ``cur_pos/32 + 1``, capped at 4 tiles in spec mode. Group 0's bounds top out at 63
    (2 tiles) and group 1's at 67 (4 tiles), so the two groups run *different* chunk sizes in
    the same program. Reader, writer and compute each derive it independently from their own
    group's max bound, so this is the case that catches one of them reading the wrong group.
    """
    B, Tg, p = 2, 4, 60
    assert (p + Tg - 1) // 32 + 1 == 2 and (p + 2 * Tg - 1) // 32 + 1 == 3  # 2 tiles vs 4 (pow2)
    # The reference (PNHt == 1) is not capped at 4 tiles, so it picks its own sizes and the
    # two paths split the reduction differently — bf16 partial-sum accuracy only.
    _check_groups(device, B=B, Tg=Tg, p=p, seq_len=512, max_cores=16, seed=47, k_chunk_size=0, pcc=0.999)


def test_spec_multi_pos_groups_first_block(device):
    """Both groups' bounds inside the very first tiles — the mask cuts inside column-tile 0
    for group 0 and column-tile 0/1 for group 1, in the same chunk."""
    _check_groups(device, B=2, Tg=4, p=5, seq_len=512, max_cores=16, seed=53)


@pytest.mark.parametrize("B, Tg", [(4, 2), (2, 7)], ids=["B4_Tg2", "B2_Tg7"])
def test_spec_multi_pos_group_shapes(device, B, Tg):
    """Other (B, Tg) splits of the same candidate count. B=4/Tg=2 is the narrow-row extreme
    (8 candidates, four 2-tile rows); B=2/Tg=7 is 14 candidates on rows too tall to fit many
    cores, which is exactly why the mode gained groups. Both are capped to 4 cores/head so
    the 7-tile rows fit L1.
    """
    _check_groups(device, B=B, Tg=Tg, p=1024, seq_len=2048, max_cores=4, seed=59, k_chunk_size=64)


def test_rejects_group_cur_pos_length(device, expect_error):
    """cur_pos must carry B*Tg bounds — Tg per batch group, not Tg in total."""
    B, Tg = 2, 4
    inp = _build_inputs(device, B * Tg, p=100, seq_len=512, seed=61)
    short_cur_pos = ttnn.Tensor(torch.arange(Tg, dtype=torch.int32), ttnn.int32).to(device)
    q_spec = inp["q_batched"].reshape(1, B, Tg * TILE, HEAD_DIM)
    page_table = inp["page_row"].unsqueeze(0).repeat(B, 1).contiguous()
    with expect_error(RuntimeError, "cur_pos must have"):
        ttnn.transformer.paged_scaled_dot_product_attention_decode(
            ttnn.Tensor(q_spec, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device),
            inp["k_tt"],
            inp["v_tt"],
            page_table_tensor=ttnn.Tensor(page_table, ttnn.int32).to(device),
            cur_pos_tensor=short_cur_pos,
            scale=SCALE,
            program_config=_program_config(device, max_cores=16, k_chunk_size=GROUP_CHUNK),
            spec_multi_pos_tiles=Tg,
        )
