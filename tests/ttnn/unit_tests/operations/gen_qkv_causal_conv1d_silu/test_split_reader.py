# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2: the ROW_MAJOR split reader (writer RISC reads body rows [32 - split_rows, 32) of
every tile-row on NoC1 and hands them to the reader through cb_split_requests / cb_split_done).

The split is pure data movement, so every split_rows value must give bit-identical outputs to the
unsplit reader (split_rows = 0). The pinned plans force the paths the default plan of the flagged
decode cells never takes: several blocks per core with block_rows >= 3 and a ragged tail (the
writer serves a whole block before its first output, plus the next block as prefetch), the DRAM
halo of a core whose range starts at t0 > 0, the L1 halo of later blocks, and an L1 input.
"""

import pytest
import torch
import ttnn

from ttnn.operations.gen_qkv_causal_conv1d_silu import get_last_plan, plan_limits
from tests.ttnn.unit_tests.operations.gen_qkv_causal_conv1d_silu.test_gen_qkv_causal_conv1d_silu import (
    PHASE0_CASES,
    _case,
    _check_case,
    _run,
)

_RM = ttnn.ROW_MAJOR_LAYOUT
_CASES = {pc.id.split("_")[0]: pc.values[0] for pc in PHASE0_CASES}
_T256 = _case("rm_T256", ((1, 256, 2560), (512, 512, 1536), 256, None, 0), _RM).values[0]


def _run_split(device, case, **limits):
    with plan_limits(**limits):
        outputs, host, device_inputs = _run(device, case)
        plan = get_last_plan()
    return outputs, host, device_inputs, plan


def test_default_plan_splits_rm_only(device):
    _, _, _, rm_plan = _run_split(device, _CASES["conv1d-001"])
    assert rm_plan.split_rows > 0
    _, _, _, tile_plan = _run_split(device, _CASES["conv1d-005"])
    assert tile_plan.split_rows == 0


@pytest.mark.parametrize(
    "case_id,limits",
    [
        pytest.param("conv1d-001", dict(), id="decode_default"),
        pytest.param("conv1d-003", dict(max_cores_per_group=2), id="dram_halo_t0_gt_0"),
        pytest.param("conv1d-006", dict(max_cores_per_group=1), id="l1_input_multiblock"),
        pytest.param("conv1d-010", dict(max_block_rows=1, max_cores_per_group=1), id="l1_output_multiblock"),
    ],
)
@pytest.mark.parametrize("split_rows", [8, 24, 32])
def test_split_matches_unsplit(device, case_id, limits, split_rows):
    case = _CASES[case_id]
    reference, host, device_inputs, plan0 = _run_split(device, case, split_rows=0, **limits)
    assert plan0.split_rows == 0
    _check_case(case, reference, host, device_inputs)
    outputs, _, _, plan = _run_split(device, case, split_rows=split_rows, **limits)
    assert plan.split_rows == split_rows
    for a, b in zip(reference, outputs):
        assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(b)), f"split_rows={split_rows} changed the result"


@pytest.mark.parametrize(
    "limits,block_rows",
    [
        # 8 tile-rows per core in blocks 3, 3, 2: the writer must serve all 3 rows of a block before
        # the block's first output exists, and prefetches the next block.
        pytest.param(dict(max_block_rows=3, max_cores_per_group=1), 3, id="block3_ragged"),
        # 2 cores per group (block_ch_tiles 2), one 4-tile-row block each: the second core's block
        # starts at t0 > 0 with a DRAM halo.
        pytest.param(dict(max_block_rows=4, max_cores_per_group=2), 4, id="block4_dram_halo"),
    ],
)
def test_split_multirow_blocks(device, limits, block_rows):
    case = _T256
    outputs, host, device_inputs, plan = _run_split(device, case, **limits)
    assert plan.regime == "rm_staged_window"
    assert plan.split_rows > 0
    assert plan.block_rows == block_rows
    _check_case(case, outputs, host, device_inputs)
    reference, _, _, _ = _run_split(device, case, split_rows=0, **limits)
    for a, b in zip(reference, outputs):
        assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(b)), "the split reader changed the result"
