# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Knob coverage for g6_permute rows_tile: plans the occupancy walk never picks on a full grid.

On a full grid only closed cells put more than one block on a core, so the direct and a-split
paths never wrap their CB rings, block_wt stays 1 and the b-split regroup (block_a > 1,
block_b < 32) never runs. These cases force those plans by turning the host knobs (max_cores,
the walk's result, BLOCK_WT, buffer depths) and check bit-exact results with the input padding
poisoned. DO NOT DELETE.
"""

import importlib

import pytest
import torch
import ttnn

from ttnn.operations.g6_permute import describe_plan, g6_permute

# The package re-exports the function g6_permute, which shadows the submodule's attribute name.
op_module = importlib.import_module("ttnn.operations.g6_permute.g6_permute")
descriptor_module = importlib.import_module("ttnn.operations.g6_permute.g6_permute_program_descriptor")

ROWS = (0, 2, 1, 3)

# id, shape, forced (block_a, block_b) or None (walk), max_cores, block_wt, max_depth_in, max_depth_staging
CASES = [
    ("direct_multiblock_wt2", (2, 3, 5, 96), (1, 1), 4, 2, 2, 2),
    ("direct_rowruns_lower_band_multiblock", (1, 5, 47, 32), (1, 4), 5, 1, 2, 2),
    ("direct_depth_in_1", (1, 6, 16, 64), (1, 2), 3, 1, 1, 2),
    ("asplit_multiblock_wt2", (1, 32, 20, 640), (8, 32), 7, 2, 2, 2),
    ("asplit16_lower_band", (1, 40, 20, 64), (16, 32), 3, 1, 2, 2),
    ("regroup_bsplit", (1, 40, 50, 64), (4, 8), 5, 1, 2, 2),
    ("regroup_bsplit_lower_band_wt3", (2, 37, 45, 96), (2, 16), 11, 3, 2, 2),
    ("closed_single_core", (3, 40, 50, 96), None, 1, 1, 2, 2),
    ("closed_staging_depth_1", (3, 40, 50, 96), None, 2, 1, 2, 1),
    ("closed_wt2_ragged", (2, 33, 65, 160), None, 3, 2, 2, 2),
]


@pytest.fixture
def forced_knobs(monkeypatch):
    def force(plan, max_cores, block_wt, max_depth_in, max_depth_staging):
        monkeypatch.setattr(descriptor_module, "max_cores_for", lambda device: max_cores)
        monkeypatch.setattr(descriptor_module, "BLOCK_WT", block_wt)
        monkeypatch.setattr(descriptor_module, "MAX_DEPTH_IN", max_depth_in)
        monkeypatch.setattr(descriptor_module, "MAX_DEPTH_STAGING", max_depth_staging)
        if plan is not None:
            monkeypatch.setattr(descriptor_module, "select_plan", lambda *args: plan)

    # A forced walk result is not part of the host cache key; start and end with empty caches.
    op_module._PLANS.clear()
    op_module._PROGRAMS.clear()
    yield force
    op_module._PLANS.clear()
    op_module._PROGRAMS.clear()


@pytest.mark.parametrize(
    "shape, plan, max_cores, block_wt, max_depth_in, max_depth_staging",
    [pytest.param(*case[1:], id=case[0]) for case in CASES],
)
def test_forced_plan_bit_exact(shape, plan, max_cores, block_wt, max_depth_in, max_depth_staging, device, forced_knobs):
    forced_knobs(plan, max_cores, block_wt, max_depth_in, max_depth_staging)
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    x_device = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    x_device = ttnn.fill_implicit_tile_padding(x_device, float("inf"), memory_config=x_device.memory_config())

    report = describe_plan(x_device, ROWS)
    if plan is not None:
        assert (report["block_a"], report["block_b"]) == plan
    assert report["block_wt"] == block_wt
    assert report["num_cores"] <= max_cores
    assert max(c["num_blocks"] for c in report["cores"]) > 1, "the case must put several blocks on a core"

    expected = torch.permute(x, ROWS).contiguous()
    actual = ttnn.to_torch(g6_permute(x_device, ROWS))
    assert list(actual.shape) == list(expected.shape)
    a = actual.to(torch.float32).view(torch.int32)
    e = expected.to(torch.float32).view(torch.int32)
    mismatch = a != e
    assert not bool(mismatch.any()), (
        f"{int(mismatch.sum())} of {mismatch.numel()} differ; first at {torch.nonzero(mismatch)[0].tolist()}; "
        f"plan {report}"
    )
