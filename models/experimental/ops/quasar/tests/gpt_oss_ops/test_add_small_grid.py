# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_add.py — do NOT regenerate this file
# from a capture.
#
# Case 03 of test_add.py adds the bias of the decode attention output projection
# (gpt_oss/tt/attention/decode.py). Its captured input, the output of that
# projection, is width-sharded in SRAM over 45 cores, up to core column 10 of the
# device the capture ran on. A Quasar device has at most 8 core columns, so
# graph_case skips the captured case on every Quasar grid.
#
# The projection asks for a width-sharded output without a shard grid, so the
# linear op picks a grid that fits the device. This variant keeps the captured call
# and width-shards the input over the first row of cores of the device under test,
# in equal tile-aligned shards; the last shard is partly empty when the width does
# not divide evenly. On the 8x4 and 2x1 Quasar grids this is the layout that
# ttnn.experimental.quasar.linear gives the projection output (checked on ttsim on
# 2026-11-23: 8 cores of 384 columns, and 2 cores of 1440 columns). The call runs on
# any grid size, so the variant does not skip.
# ---------------------------------------------------------------------------
"""Small-grid companion of the ``ttnn.add`` bias add after the decode attention output projection (case 03)."""

import copy
import math

import pytest

import ttnn
from models.experimental.ops.quasar.tests.gpt_oss_ops import graph_case as G
from models.experimental.ops.quasar.tests.gpt_oss_ops import test_add as T


def _input_on_first_core_row(case, mesh_device):
    """The captured case with its activation (args[0]) width-sharded over the first core row of the device."""
    small = copy.deepcopy(case)
    small["count"] = 1
    activation = small["args"][0]
    rows, width = activation["shape"][-2], activation["shape"][-1]
    num_cores = mesh_device.compute_with_storage_grid_size().x
    shard_width = math.ceil(width / ttnn.TILE_SIZE / num_cores) * ttnn.TILE_SIZE
    activation["mem"]["shard"] = {
        "grid": [[0, 0, num_cores - 1, 0]],
        "shape": [rows, shard_width],
        "orientation": "ROW_MAJOR",
    }
    return small


CASES = [c for c in T.CASES if c["id"] == "03_32x2880_bf16_ws-l1"]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] + "-first_core_row" for c in CASES])
def test_add_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(T._OP, _input_on_first_core_row(case, ttnn_mesh_device), ttnn_mesh_device)
