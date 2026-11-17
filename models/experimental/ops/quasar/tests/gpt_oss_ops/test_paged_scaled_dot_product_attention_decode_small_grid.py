# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated
# test_paged_scaled_dot_product_attention_decode.py — do NOT regenerate this file
# from a capture.
#
# Both captured cases pin SDPAProgramConfig.compute_with_storage_grid_size = [8, 4]
# (32 cores), the full grid of the device the capture ran on. On a device with a
# smaller compute grid, sdpa_decode_program_factory.cpp FATALs with "Cores available
# (32) exceeds grid size (N)".
#
# The gpt-oss model does not pin this grid: it builds the decode SDPAProgramConfig
# with get_compute_grid_size(mesh_device) (gpt_oss/tt/attention/config.py), which is
# the device grid, limited to 8x8. On a 2x1 device the model therefore asks for a
# 2x1 grid. Each variant below is a captured case with only the program config's
# compute grid replaced by that 2x1 grid; inputs, dtypes, layouts, memory configs,
# sliding window, attention sink, scale and chunk sizes stay as captured.
#
# With batch 1 and 8 KV heads, the factory's core allocation puts 4 KV heads on each
# of the 2 cores (on the captured 8x4 grid it uses 4 cores per KV head instead), so
# these variants cover the several-heads-per-core path. graph_case has no torch
# reference for this op, so run_case checks the output spec and finite values only.
#
# The captured Q tensor is height-sharded on the single core (0, 0), which exists on
# any device, so the Q shard grid needs no change.
#
# Each variant SKIPS (rather than FATALs) when the device compute grid is smaller
# than the variant's grid rectangle, so the file is safe to run on any device.
# ---------------------------------------------------------------------------
"""Small-grid (2x1) variants of ``paged_scaled_dot_product_attention_decode``, as the gpt-oss model runs it."""

import copy

import pytest

from models.experimental.ops.quasar.tests.gpt_oss_ops import graph_case as G
from models.experimental.ops.quasar.tests.gpt_oss_ops.test_paged_scaled_dot_product_attention_decode import (
    CASES as _CAPTURED_CASES,
    _OP,
)

# The compute grid get_compute_grid_size() returns on a 2x1 device.
_SMALL_GRID = (2, 1)


def _case_with_grid(case, grid_xy):
    small = copy.deepcopy(case)
    small["id"] = f"{case['id']}_{grid_xy[0]}x{grid_xy[1]}grid"
    small["kwargs"]["program_config"]["fields"]["compute_with_storage_grid_size"] = list(grid_xy)
    return small


CASES = [_case_with_grid(c, _SMALL_GRID) for c in _CAPTURED_CASES]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_paged_scaled_dot_product_attention_decode_small_grid(ttnn_mesh_device, reset_seeds, case):
    grid_x, grid_y = case["kwargs"]["program_config"]["fields"]["compute_with_storage_grid_size"]
    dev = ttnn_mesh_device.compute_with_storage_grid_size()
    if dev.x < grid_x or dev.y < grid_y:
        pytest.skip(f"case needs a {grid_x}x{grid_y} grid rectangle; device compute grid is {dev.x}x{dev.y}")
    G.run_case(_OP, case, ttnn_mesh_device)
