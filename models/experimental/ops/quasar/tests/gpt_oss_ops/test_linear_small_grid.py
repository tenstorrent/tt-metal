# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_linear.py — do NOT regenerate this file
# from a capture.
#
# Case 00 of test_linear.py is the output projection of decode attention
# (gpt_oss/tt/attention/decode.py). Its captured input, the merged heads, is
# width-sharded in SRAM over 64 cores, up to core column 10 of the device the
# capture ran on. A Quasar device has at most 8 core columns, so graph_case skips
# the captured case on every Quasar grid.
#
# On a device with fewer cores than heads, the model merges the heads with
# merge_heads_on_device instead of ttnn.experimental.nlp_concat_heads_decode, and
# merge_heads_on_device returns the merged heads interleaved in DRAM. This variant
# keeps the captured call (weight, dtype, and the width-sharded output memory config
# without a shard grid, so the op picks a grid that fits the device) and moves only
# the input to interleaved DRAM. The captured output shard grid does not exist on a
# smaller device, so graph_case checks only the output layout and buffer type there.
# The call runs on any grid size, so the variant does not skip.
# ---------------------------------------------------------------------------
"""Small-grid companion of the ``ttnn.linear`` decode attention output projection (case 00)."""

import copy

import pytest

from models.experimental.ops.quasar.tests.gpt_oss_ops import graph_case as G
from models.experimental.ops.quasar.tests.gpt_oss_ops import test_linear as T

_DRAM = {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None}


def _input_in_dram(case):
    """The captured case with its activation (args[0]) interleaved in DRAM."""
    small = copy.deepcopy(case)
    small["id"] = case["id"].replace("_ws-l1", "_int-dram", 1)
    small["count"] = 1
    small["args"][0]["mem"] = dict(_DRAM)
    return small


CASES = [_input_in_dram(c) for c in T.CASES if c["id"] == "00_32x4096_bf16_ws-l1"]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_linear_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(T._OP, case, ttnn_mesh_device)
