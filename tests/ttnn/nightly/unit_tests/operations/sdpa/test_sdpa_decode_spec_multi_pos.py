# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Nightly sweep for ``spec_multi_pos_tiles`` on
``ttnn.transformer.paged_scaled_dot_product_attention_decode``.

Post-commit smoke and validation tests: tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode_spec_multi_pos.py.

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

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.unit_tests.operations.sdpa.spec_multi_pos_test_utils import (
    build_inputs,
    check,
    check_bit_exact,
    check_groups,
    check_groups_bit_exact,
    config_for,
    program_config,
    run_reference,
    run_spec,
    skip_if_exceeds_l1,
    torch_reference,
)

# One module-scoped device: the KV caches below are up to 32k tokens and every case
# reuses the same two programs per (T, seq_len).
pytestmark = pytest.mark.use_module_device

# ── Equivalence sweep ──────────────────────────────────────────────────────
#
# ``p`` walks tile-alignment edges (p % 32 in {0, 31}) and k-chunk edges. The chunk
# width differs per T (see SPEC_CONFIG: 128 for T=4, 64 for T=7, 32 for T=11), so each p
# lands on a different point of each T's chunk grid — which is the intent: between them the
# cases cover "bound is the last position of a chunk" (one masked chunk), "bounds straddle a
# chunk boundary" (two masked chunks, possibly on two different cores) and "bounds sit well
# inside a chunk".


@pytest.mark.parametrize("T", [4, 7, 11], ids=["T4", "T7", "T11"])
@pytest.mark.parametrize(
    "seq_len, p",
    [
        # 2k context, tile-alignment edges (p % 32 in {0, 31})
        (2048, 1024),  # p % 32 == 0,  p % 128 == 0
        (2048, 1055),  # p % 32 == 31
        # chunk-boundary edges
        (2048, 1023),  # last position of a 128-chunk: bounds start a fresh chunk
        (2048, 1021),  # bounds straddle a 128-chunk boundary -> TWO masked chunks
        # 8k context
        (8192, 4093),  # straddles a chunk boundary
    ],
    ids=[
        "s2k_p1024",
        "s2k_p1055",
        "s2k_p1023",
        "s2k_p1021_straddle",
        "s8k_p4093_straddle",
    ],
)
def test_spec_multi_pos_matches_batched(device, T, seq_len, p):
    skip_if_exceeds_l1(T)
    torch.manual_seed(0)
    assert p + T - 1 < seq_len
    check(device, T, p, seq_len)


@pytest.mark.parametrize("T", [4, 7, 11], ids=["T4", "T7", "T11"])
@pytest.mark.parametrize("seq_len, p", [(2048, 1024), (8192, 5000)], ids=["s2k", "s8k"])
def test_spec_multi_pos_is_bit_exact(device, T, seq_len, p):
    """The sharpest form of the claim. With the reduction tree matched (SPEC_CONFIG) and the
    T bounds inside a single k-chunk, folding the candidates onto one batch row is not merely
    numerically close to the B == T form — it produces the identical bits.
    """
    skip_if_exceeds_l1(T)
    torch.manual_seed(5)
    check_bit_exact(device, T, p, seq_len)


@pytest.mark.parametrize("T", [4, 7], ids=["T4", "T7"])
@pytest.mark.parametrize("p", [30000, 32639], ids=["p30000", "p32639_straddle"])
def test_spec_multi_pos_long_context(device, T, p):
    """32k context — the regime the mode exists for (DRAM-bound KV scan)."""
    torch.manual_seed(1)
    check(device, T, p, seq_len=32768)


@pytest.mark.parametrize("p", [30000, 32639], ids=["p30000", "p32639_straddle"])
def test_spec_multi_pos_long_context_single_core(device, p):
    """T=11 at 32k. Only 1 core/head fits L1 at T=11 (see SPEC_CONFIG), so this run
    accumulates ~1000 chunks of bf16 flash-decode state serially on one core. That costs
    absolute accuracy — but it costs the LEGACY path exactly as much: at this config both
    paths land at ~0.988 PCC against fp32, while the same context with a legacy-friendly
    config (8 cores/head, 128-wide chunks) reaches 0.9996. So the assertion here is the
    equivalence itself, checked against the batched reference rather than against torch.
    """
    T = 11
    skip_if_exceeds_l1(T)
    torch.manual_seed(2)
    inp = build_inputs(device, T, p, seq_len=32768, seed=17)
    pc = config_for(device, T)
    ref = run_reference(device, inp, T, pc)
    spec = run_spec(device, inp, T, pc)
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
    inp = build_inputs(device, T, p, seq_len, seed=19)
    pc = program_config(device, max_cores=64, k_chunk_size=128)
    ref = run_reference(device, inp, T, pc)
    spec = run_spec(device, inp, T, pc)
    torch_ref = torch_reference(inp["q_heads"], inp["k"], inp["v"], inp["cur_pos"])

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
    check(device, T=4, p=40, seq_len=512, seed=7)


def test_spec_multi_pos_first_block(device):
    """cur_pos inside the very first tile — the mask cuts inside column-tile 0."""
    torch.manual_seed(2)
    check(device, T=7, p=5, seq_len=512, seed=11)


def test_spec_multi_pos_dynamic_chunk(device):
    """``k_chunk_size=0``: the k-chunk is picked in-kernel from cur_pos and capped at 4 tiles
    in spec mode. The reference (PNHt == 1) is not capped and picks 8 tiles, so the two paths
    split the flash reduction differently and only agree to bf16 partial-sum accuracy — hence
    the looser bound here than in the matched-split sweep above. Both are still held to the
    fp32 torch reference at the usual 0.99."""
    torch.manual_seed(3)
    pc = program_config(device, max_cores=64, k_chunk_size=0)
    check(device, T=4, p=1024, seq_len=2048, seed=13, program_config=pc, pcc_ref_vs_spec=0.999)


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
    check_groups(device, B=2, Tg=4, p=p, seq_len=seq_len, max_cores=55, seed=31)


@pytest.mark.parametrize(
    "seq_len, p",
    [(2048, 1024), (2048, 1021), (8192, 5000)],
    ids=["s2k", "s2k_straddle", "s8k"],
)
def test_spec_multi_pos_groups_16_cores(device, seq_len, p):
    """The same claim at a much narrower split (16 cores/head), where each core carries many
    more chunks and the two masked chunks are far more likely to land on the same core."""
    check_groups(device, B=2, Tg=4, p=p, seq_len=seq_len, max_cores=16, seed=37)


@pytest.mark.parametrize("seq_len, p", [(2048, 1024), (8192, 5000), (32768, 30000)], ids=["s2k", "s8k", "s32k"])
def test_spec_multi_pos_groups_is_bit_exact(device, seq_len, p):
    """The sharpest form of the grouped claim. ``max_cores_per_head_batch=8`` puts the B=2 spec
    call and the B=8 reference on the same cores-per-head (both are capped by max_cores, not by
    the grid), so the reduction trees match; with no group straddling a chunk boundary the two
    forms are not merely close but identical bit-for-bit.
    """
    check_groups_bit_exact(device, seq_len, p)


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
    check_groups(device, B=B, Tg=Tg, p=p, seq_len=512, max_cores=16, seed=47, k_chunk_size=0, pcc=0.999)


def test_spec_multi_pos_groups_first_block(device):
    """Both groups' bounds inside the very first tiles — the mask cuts inside column-tile 0
    for group 0 and column-tile 0/1 for group 1, in the same chunk."""
    check_groups(device, B=2, Tg=4, p=5, seq_len=512, max_cores=16, seed=53)


@pytest.mark.parametrize("B, Tg", [(4, 2), (2, 7)], ids=["B4_Tg2", "B2_Tg7"])
def test_spec_multi_pos_group_shapes(device, B, Tg):
    """Other (B, Tg) splits of the same candidate count. B=4/Tg=2 is the narrow-row extreme
    (8 candidates, four 2-tile rows); B=2/Tg=7 is 14 candidates on rows too tall to fit many
    cores, which is exactly why the mode gained groups. Both are capped to 4 cores/head so
    the 7-tile rows fit L1.
    """
    check_groups(device, B=B, Tg=Tg, p=1024, seq_len=2048, max_cores=4, seed=59, k_chunk_size=64)


@pytest.mark.parametrize(
    "seq_len, p",
    [
        (2048, 1024),
        (2048, 1020),
        (2048, 1021),
        (32768, 30000),
    ],
    ids=[
        "s2k_p1024",
        "s2k_p1020_split_scan",
        "s2k_p1021_straddle",
        "s32k_p30000",
    ],
)
@pytest.mark.parametrize("k_chunk_size", [0, 128], ids=["B3_Tg4_dyn_chunk", "B3_Tg4_chunk128"])
def test_spec_groups_b3_tg4_full_grid(device, seq_len, p, k_chunk_size):
    """B=3 groups of Tg=4 (12 candidates) at 36 cores/head — the T=12 full-grid config the model
    uses (``_SPEC_SDPA_L1_FIT[12]``): three ~36-core reduction groups, 108 of ~110 cores active,
    the KV read three times for 12 candidates instead of twelve. Same Tg=4 CB footprint as the
    B=2 config; the only new thing is a third group and a reduction tree at the 6-round cap.

    Run once with the in-kernel dynamic chunk (what the model passes) and once at the fixed
    4-tile chunk. The reference (B=12 rows) is capped by its own row count to ~9 cores/head, so
    the reduction trees differ and only bf16 partial-sum accuracy is available (pcc pinned
    explicitly: the default's ``group_straddles`` cannot be evaluated at ``k_chunk_size=0``).
    """
    check_groups(device, B=3, Tg=4, p=p, seq_len=seq_len, max_cores=36, seed=67, k_chunk_size=k_chunk_size, pcc=0.999)
