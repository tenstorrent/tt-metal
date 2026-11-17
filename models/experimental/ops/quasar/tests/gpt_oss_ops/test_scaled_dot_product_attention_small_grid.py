# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_scaled_dot_product_attention.py — do NOT
# regenerate this file from a capture.
#
# Both captured cases set SDPAProgramConfig.compute_with_storage_grid_size = [8, 4]
# (32 cores), because the capture ran on an 8x4 device. On a device with a smaller
# compute grid (for example the 2x1 grid of the 2-core Quasar emulator) the op
# FATALs with "Provided grid must not contain more cores than the device".
#
# The gpt-oss model does not use a fixed grid for SDPA: its program configs take
# compute_with_storage_grid_size from get_compute_grid_size()
# (models/experimental/ops/quasar/gpt_oss/tt/attention/config.py), which is the
# device grid limited to 8x8. On a 2x1 device the model therefore passes a 2x1 grid.
#
# The only change from each captured case is compute_with_storage_grid_size = [2, 1]
# (2 cores), the grid the model would pick on a 2x1 device. The shapes, dtypes,
# layouts, memory configs, sliding window, attention sink and chunk sizes are the
# captured ones. A 32-core variant is not included, because the 32-core grid is the
# captured grid and the generated test already runs it.
#
# Each variant SKIPS (rather than FATALs) when the device compute grid is smaller
# than the variant's grid rectangle, so the file is safe to run on any device.
# ---------------------------------------------------------------------------
"""Small-grid (2-core) variants of ``scaled_dot_product_attention`` (prefill)."""

import copy

import pytest

import ttnn
from models.experimental.ops.quasar.tests.gpt_oss_ops import graph_case as G
from models.experimental.ops.quasar.tests.gpt_oss_ops.test_scaled_dot_product_attention import CASES as _CAPTURED

_OP = ttnn.experimental.quasar.transformer.scaled_dot_product_attention


def _case_with_grid(base, grid_xy):
    case = copy.deepcopy(base)
    grid_x, grid_y = grid_xy
    case["id"] = f"{base['id']}-{grid_x * grid_y}core_{grid_x}x{grid_y}"
    case["kwargs"]["program_config"]["fields"]["compute_with_storage_grid_size"] = [grid_x, grid_y]
    return case


CASES = [_case_with_grid(c, (2, 1)) for c in _CAPTURED]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_scaled_dot_product_attention_small_grid(ttnn_mesh_device, reset_seeds, case):
    grid_x, grid_y = case["kwargs"]["program_config"]["fields"]["compute_with_storage_grid_size"]
    dev = ttnn_mesh_device.compute_with_storage_grid_size()
    if dev.x < grid_x or dev.y < grid_y:
        pytest.skip(f"case needs a {grid_x}x{grid_y} grid rectangle; device compute grid is {dev.x}x{dev.y}")
    G.run_case(_OP, case, ttnn_mesh_device)
