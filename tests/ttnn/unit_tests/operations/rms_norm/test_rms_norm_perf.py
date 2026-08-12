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
