# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-kernel perf harness for rms_norm — DO NOT DELETE.

Shapes and reference latencies come from
`eval/golden_tests/rms_norm/feature_spec.py`'s `perf` loose cases
(`achievable_ns`, measured on blackhole_p150b at 1350 MHz).  Those loose cases
pin `fp32_dest_acc_en=False` + `math_fidelity=HiFi2`; Refinement 1 added
`fp32_dest_acc_en=False` to SUPPORTED, so since then this harness runs the
**exact** pinned perf configuration (it used to proxy it at
`fp32_dest_acc_en=True`, because that value was outside Phase 0's rectangle).

Run under the profiler for DEVICE KERNEL DURATION [ns]:

    scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_perf.py

Correctness is still asserted here; the reference latencies are recorded in the
ids, not gated, because the measurement lives in the profiler CSV.
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm


#          rows,  hidden, achievable_ns (interleaved, blackhole_p150b @1350MHz)
PERF_CASES = [
    (32, 1024, 9149),
    (32, 2304, 17003),
    (32, 5120, 75825),
    (32, 7168, 104259),  # requires >= 7x -> <= 14894 ns
    (8192, 1024, 96744),
    (8192, 2304, 211345),
    (8192, 5120, 738307),
    (8192, 7168, 1032281),
]


@pytest.mark.parametrize(
    "rows,hidden,achievable_ns",
    PERF_CASES,
    ids=[f"r{r}_h{h}_ref{n}ns" for r, h, n in PERF_CASES],
)
def test_rms_norm_perf(device, rows, hidden, achievable_ns):
    torch.manual_seed(0)
    shape = (1, 1, rows, hidden)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    torch_gamma = torch.randn((1, 1, 1, hidden), dtype=torch.float32).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_gamma = ttnn.from_torch(torch_gamma, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
    tt_out = rms_norm(tt_input, gamma=tt_gamma, epsilon=1e-6, compute_kernel_config=cfg)

    x = torch_input.float()
    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * torch_gamma.float().reshape(-1)
    actual = ttnn.to_torch(tt_out).float()
    from tests.ttnn.utils_for_testing import assert_with_pcc

    assert_with_pcc(expected, actual, 0.995)


# --- Refinement 5: the five pinned SHARDED perf geometries -------------------
#
# feature_spec's `group="perf"` sharded loose cases. Their `achievable_ns` are the
# measured latency of THAT geometry (2-20x tighter than the interleaved twin), so
# they may only ever be compared against a sharded measurement at the same pinned
# config (bf16 / TILE / fp32_dest_acc_en=False / HiFi2) and the same pinned shard
# spec — never against an interleaved number.
#
#          rows, hidden, achievable_ns, placement,      shard[h, w],   grid(x, y)
SHARD_PERF_CASES = [
    (32, 1024, 4110, "WIDTH", [32, 128], (8, 1)),
    (32, 2304, 4617, "WIDTH", [32, 256], (9, 1)),
    (32, 5120, 5267, "WIDTH", [32, 160], (8, 4)),
    (32, 7168, 5481, "WIDTH", [32, 256], (7, 4)),
    (8192, 1024, 25640, "BLOCK", [1024, 128], (8, 8)),
]

_SHARD_ML = {
    "WIDTH": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    "BLOCK": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    "HEIGHT": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
}


def _run_sharded_perf(device, rows, hidden, scheme, shard_shape, core_grid):
    from eval.sharding import shard_config

    torch.manual_seed(0)
    shape = (1, 1, rows, hidden)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    torch_gamma = torch.randn((1, 1, 1, hidden), dtype=torch.float32).to(torch.bfloat16)

    memory_config = shard_config(
        shard_shape,
        core_grid,
        _SHARD_ML[scheme],
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    tt_gamma = ttnn.from_torch(torch_gamma, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
    tt_out = rms_norm(
        tt_input, gamma=tt_gamma, epsilon=1e-6, compute_kernel_config=cfg, memory_config=tt_input.memory_config()
    )

    x = torch_input.float()
    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * torch_gamma.float().reshape(-1)
    from tests.ttnn.utils_for_testing import assert_with_pcc

    assert_with_pcc(expected, ttnn.to_torch(tt_out).float(), 0.995)


@pytest.mark.parametrize(
    "rows,hidden,achievable_ns,scheme,shard_shape,core_grid",
    SHARD_PERF_CASES,
    ids=[f"{s.lower()}_r{r}_h{h}_ref{n}ns" for r, h, n, s, _, _ in SHARD_PERF_CASES],
)
def test_rms_norm_perf_sharded(device, rows, hidden, achievable_ns, scheme, shard_shape, core_grid):
    _run_sharded_perf(device, rows, hidden, scheme, shard_shape, core_grid)


# --- Refinement 5 diagnostic: how much of the BLOCK profile is the COMBINE? ---
#
# `(1,1,8192,1024)` BLOCK [1024,128] on (8,8) gives every core 32 tile-rows x 4
# hidden tiles = 128 tiles and a G=8 cross-core reduction group per grid row.
# The HEIGHT twin below puts the SAME 128 tiles on each of the SAME 64 cores
# (4 tile-rows x 32 hidden tiles) with G == 1 — no gather, no multicast, no root
# reduce. The pair isolates the combine at equal compute and equal residency.


@pytest.mark.parametrize(
    "scheme,shard_shape,core_grid",
    [
        ("BLOCK", [1024, 128], (8, 8)),
        ("HEIGHT", [128, 1024], (8, 8)),
    ],
    ids=["block_g8", "height_g1"],
)
def test_rms_norm_perf_sharded_combine_share(device, scheme, shard_shape, core_grid):
    _run_sharded_perf(device, 8192, 1024, scheme, shard_shape, core_grid)


# --- Refinement 5: the block count of a sharded cross-core combine ------------
#
# MAX_GATHER_TILES caps `R * G` — i.e. cb_stat_gather's page count — and on the
# BLOCK perf geometry (G = 8, 32 tile-rows per core) it is what sets R = 8 and so
# nblocks = 4. Every block is a full group-wide barrier (gather -> root reduce ->
# finalize -> multicast) plus a fresh set of phase-boundary LLK inits, so sweeping
# the cap measures the PER-BLOCK cost directly: duration against nblocks.


@pytest.mark.parametrize("gather_cap", [16, 32, 64, 128, 256], ids=lambda c: f"cap{c}")
def test_rms_norm_perf_sharded_gather_cap(device, gather_cap, monkeypatch):
    from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

    monkeypatch.setattr(pd, "MAX_GATHER_TILES", gather_cap)
    _run_sharded_perf(device, 8192, 1024, "BLOCK", [1024, 128], (8, 8))


# --- Refinement 5 diagnostic: the combine at MATCHED (C, R, nblocks) ----------
#
# The HEIGHT twin above removes the combine but also changes C (4 -> 32) and the
# block count (4 -> 1), so its 25.5 us floor mixes the combine with the block
# schedule. This forces the BLOCK spec's groups to ONE member each — same shard,
# same C = 4, same R = 8, same 4 blocks, no gather / root reduce / multicast.
# NUMERICALLY WRONG on purpose (each core normalises by its own quarter row); it
# is a measurement, so it is `nogather` and skipped unless selected explicitly.


@pytest.mark.parametrize("nogather", [False, True], ids=["combine", "nogather"])
def test_rms_norm_perf_sharded_nogather(device, nogather, monkeypatch):
    from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

    monkeypatch.setattr(pd, "MAX_GATHER_TILES", 64 if not nogather else 8)
    if nogather:
        real = pd._sharded_groups

        def per_core_groups(sv, infos):
            out = []
            for g in real(sv, infos):
                for m in g["members"]:
                    out.append({**g, "members": [m], "box_cores": [m["core"]]})
            return out

        monkeypatch.setattr(pd, "_sharded_groups", per_core_groups)
    try:
        _run_sharded_perf(device, 8192, 1024, "BLOCK", [1024, 128], (8, 8))
    except AssertionError:
        if not nogather:
            raise  # the un-ablated run must still be correct


# --- perf lamp P2 sweep: cap the cores per reduction group -------------------
#
# Maximum occupancy is the selection function's default; at tensor_row_tiles == 1
# it pushes w_group_size to the whole grid, so a decode shape pays a full
# gather + multicast round over 110 cores for 3-4 hidden tiles of real work per
# core. The measured-fastest geometries in feature_spec.py for these shapes use
# 28-32 cores, so sweep the cap and read the winner off the profiler CSV.


@pytest.mark.parametrize("cap", [0, 8, 16, 32, 64], ids=lambda c: f"cap{c}")
@pytest.mark.parametrize(
    "rows,hidden",
    [(32, 1024), (32, 2304), (32, 5120), (32, 7168), (8192, 1024), (8192, 5120)],
    ids=lambda v: str(v),
)
def test_rms_norm_perf_wgroup_cap(device, rows, hidden, cap, monkeypatch):
    from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

    monkeypatch.setattr(pd, "MAX_W_GROUP_SIZE", cap)
    test_rms_norm_perf(device, rows, hidden, 0)


# --- perf lamp P1 sweep: block count vs read/compute overlap ------------------
#
# input_cb_depth = 2 only buys overlap when there is a block b+1 for the reader
# to prefetch. At the coarsest block_row_tiles a prefill core often gets exactly
# one block, so the whole DRAM read serializes against compute. Sweep the
# minimum block count and read the winner off the profiler CSV.


@pytest.mark.parametrize("min_blocks", [1, 2, 3, 4], ids=lambda b: f"blocks{b}")
@pytest.mark.parametrize(
    "rows,hidden",
    [(8192, 1024), (8192, 2304), (8192, 5120), (8192, 7168), (32, 7168)],
    ids=lambda v: str(v),
)
def test_rms_norm_perf_pipeline_blocks(device, rows, hidden, min_blocks, monkeypatch):
    from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

    monkeypatch.setattr(pd, "MIN_PIPELINE_BLOCKS", min_blocks)
    test_rms_norm_perf(device, rows, hidden, 0)


# --- Refinement 4: collateral shapes of the critical-path admissibility band ------
#
# `_admissible_by_balance` (BALANCE_SLACK_PCT + MIN_CORE_W_TILES) changes the chosen
# (G, C, R) for exactly four of the 31 shapes surveyed in probes/probe_030.py: the
# two prefill perf cases it targets, plus these two. They are measured here so the
# band's blast radius is covered by numbers rather than by argument.


@pytest.mark.parametrize(
    "shape",
    [(3, 1, 736, 5119), (1, 1, 4096, 4096)],
    ids=lambda s: "x".join(str(d) for d in s),
)
@pytest.mark.parametrize("banded", [False, True], ids=["phase0", "banded"])
def test_rms_norm_perf_balance_collateral(device, shape, banded, monkeypatch):
    from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

    if not banded:
        monkeypatch.setattr(pd, "BALANCE_SLACK_PCT", None)

    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    torch_gamma = torch.randn((shape[-1],), dtype=torch.float32).to(torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_gamma = ttnn.from_torch(torch_gamma, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
    tt_out = rms_norm(tt_input, gamma=tt_gamma, epsilon=1e-6, compute_kernel_config=cfg)

    x = torch_input.float()
    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * torch_gamma.float()
    from tests.ttnn.utils_for_testing import assert_with_pcc

    assert_with_pcc(expected, ttnn.to_torch(tt_out).float(), 0.995)


# --- Refinement 4: reduction-group FLOOR (op_design.md lamp P3) ------------------
#
# A prefill shape fills the grid on the `row` axis alone, so the selection prefers
# G = 1 (fewest combine partners). But `row` splits in whole tile-rows, and 256
# tile-rows over 110 groups is 3-vs-2: the critical core carries 1.29x the mean.
# Splitting `hidden` as well multiplies the number of tile-rows per group and so
# QUANTISES the row split more finely (G=2 -> 55 groups -> 5-vs-4 -> 1.07x), at the
# cost of one combine round per block and a narrower per-core DRAM run.
#
# Sweep the floor to find where that trade turns. MIN_W_GROUP_SIZE is a preference,
# so the selection still takes the SMALLEST G >= the floor that fills the grid.


@pytest.mark.parametrize("min_g", [1, 2, 5, 10, 11], ids=lambda g: f"ming{g}")
@pytest.mark.parametrize("rows,hidden", [(8192, 1024), (8192, 2304), (8192, 5120), (8192, 7168)], ids=lambda v: str(v))
def test_rms_norm_perf_wgroup_min(device, rows, hidden, min_g, monkeypatch):
    from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

    monkeypatch.setattr(pd, "MIN_W_GROUP_SIZE", min_g)
    test_rms_norm_perf(device, rows, hidden, 0)


# --- Refinement 4: buffer-depth co-tune (op_design.md lamp P1, `double_buffer`) --
#
# input_cb_depth and output_cb_depth trade the SAME L1 as block_row_tiles (the
# residency solve spends what is left on R), so they have to be swept together
# rather than stacked blind. Depth buys bytes-in-flight-per-barrier and
# read/compute/write overlap; R buys combine rounds amortised.


@pytest.mark.parametrize("in_depth,out_depth", [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (4, 4)])
@pytest.mark.parametrize("rows,hidden", [(8192, 1024), (8192, 2304), (8192, 5120), (8192, 7168)], ids=lambda v: str(v))
def test_rms_norm_perf_cb_depths(device, rows, hidden, in_depth, out_depth, monkeypatch):
    from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

    monkeypatch.setattr(pd, "INPUT_CB_DEPTH", in_depth)
    monkeypatch.setattr(pd, "OUTPUT_CB_DEPTH", out_depth)
    test_rms_norm_perf(device, rows, hidden, 0)


# --- Refinement 4: is the prefill wall DRAM bandwidth, or the row imbalance? ---
#
# The prefill profile is aggregate-DRAM-bandwidth bound, so the interesting
# question is what fraction of the wall is the SELECTION's tile-row imbalance
# rather than the DRAM itself: 256 tile-rows over 110 row-groups is 36 groups with
# 3 and 74 with 2, so the critical-path core carries 3/(256/110) = 1.29x the
# average and the last third of the op runs with only 36 of 110 cores demanding
# DRAM.
#
# This sweep isolates that WITHOUT changing any code: each shape below is the same
# hidden size at a row count that the SAME split divides EXACTLY (a perfectly
# balanced twin), next to the 8192-row case that does not. Divide bytes by the
# measured device-kernel ns to compare achieved GB/s directly.
#
#   W=1024 -> G=1, 110 row-groups: balanced at rows % (110*32) == 0
#   W=7168 -> G=2,  55 row-groups: balanced at rows % ( 55*32) == 0


@pytest.mark.parametrize(
    "rows,hidden",
    [
        (7040, 1024),  # 220 tile-rows / 110 groups = 2 each, exact
        (8192, 1024),  # 256 tile-rows / 110 groups = 3-vs-2  <-- the perf case
        (10560, 1024),  # 330 tile-rows / 110 groups = 3 each, exact
        (8800, 7168),  # 275 tile-rows /  55 groups = 5 each, exact
        (8192, 7168),  # 256 tile-rows /  55 groups = 5-vs-4  <-- the perf case
    ],
    ids=lambda v: str(v),
)
def test_rms_norm_perf_row_balance(device, rows, hidden):
    test_rms_norm_perf(device, rows, hidden, 0)


# --- Refinement 3: the combine tree is structural, not numerical ---------------
#
# The two-stage grid combine and the flat root-gather produce the SAME numbers, so
# no value check can tell them apart — only the descriptor can (the same argument
# as the zero-copy shard assertion and the chunking knob in test_rms_norm_sharded).
# Pin both directions: a 2-D reduction group must take the tree, and a group that
# is a single grid row (or has a level-1 span of 1) must keep the flat path, which
# is the Phase 0 code verbatim.


def test_rms_norm_combine_tree_shape():
    """`_tree_for_box` — level-1 x level-2 fan-in of the combine, per group box."""
    from ttnn.operations.rms_norm.rms_norm_program_descriptor import _tree_for_box

    # A fully populated multi-row rectangle reduces along x, then along y.
    assert _tree_for_box(22, 11, 2) == (11, 2)
    assert _tree_for_box(110, 11, 10) == (11, 10)
    # One grid row: nothing to do at level 2.
    assert _tree_for_box(11, 11, 1) == (11, 1)
    # A vertical line: level 1 would be a self-write, so stay flat (an extra hop
    # costs ~1 us of NoC + semaphore + CB latency and folds the same tiles).
    assert _tree_for_box(5, 1, 5) == (5, 1)
    # A box with filler cores (a shard grid that is not a rectangle): flat, because
    # a leader's ACTIVE row count is not group-uniform.
    assert _tree_for_box(16, 11, 2) == (16, 1)
    # The degenerate single-core group.
    assert _tree_for_box(1, 1, 1) == (1, 1)


@pytest.mark.parametrize(
    "rows,hidden,expect_two_stage",
    [
        # tensor_row_tiles == 1 fills the grid along `hidden`, so the group is the
        # whole 2-D grid -> tree.
        (32, 7168, True),
        # tensor_row_tiles >> num_groups keeps w_group_size == 1 -> flat.
        (8192, 1024, False),
    ],
    ids=["decode_two_stage", "prefill_flat"],
)
def test_rms_norm_combine_tree_is_selected(device, rows, hidden, expect_two_stage):
    from ttnn.operations.rms_norm.rms_norm_program_descriptor import (
        create_program_descriptor,
        WRITER_MCAST_CT_BASE,
    )

    shape = (1, 1, rows, hidden)
    tt_input = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_gamma = ttnn.from_torch(
        torch.zeros((1, 1, 1, hidden), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, tt_input.memory_config()
    )
    descriptor = create_program_descriptor(
        tt_input, tt_gamma, tt_out, epsilon=1e-6, compute_kernel_config=ttnn.ComputeConfigDescriptor()
    )
    # writer CT layout ends [..., w_chunk_tiles, STAGE2_SPAN, SEM_GATHER2] before
    # the mcast block — see WRITER_MCAST_CT_BASE in the program descriptor.
    stage2_span = list(descriptor.kernels[1].compile_time_args)[WRITER_MCAST_CT_BASE - 2]
    assert (stage2_span > 1) == expect_two_stage, f"stage2_span={stage2_span} for {shape}"
