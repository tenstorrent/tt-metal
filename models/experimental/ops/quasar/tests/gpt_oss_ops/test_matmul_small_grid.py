# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_matmul.py — do NOT regenerate this file
# from a capture.
#
# Cases 03 and 04 of test_matmul.py are the lm_head matmuls: [1,2880] and
# [32,2880] activations times the [2880,262144] weight. On the full 8x4 grid of
# the simulated device each takes about 6.5 min on craq-sim. On a 2x1 grid
# (2 worker cores) the same work goes to 16x fewer cores, and craq-sim runs it
# serially, so each case runs for much longer than the 15 min a suite case gets.
# The op does not hang: run alone on a 2x1 grid on 2026-11-12, case 03 passed in
# 39 min (114M simulated cycles) and case 04 in 31 min (75M cycles). Nothing in
# either call needs the full vocabulary width.
#
# These variants keep each captured call (dtypes, layouts, memory configs, kwargs)
# and change only the weight width, from 262144 to _W = 2048 (64 tiles), so the
# cases finish in a few minutes on 2 cores. At _W the inputs (about 6.3 MB) also
# fit the input-size limit of the emulator subset in conftest.py.
#
# On a 2-core grid, ttnn.matmul picks the same program config for these calls as
# for the captured calls (create_simple_matmul_program_config in
# matmul_program_config.cpp):
#   * mirrors 03 (activation in SRAM): the output splits into blocks of up to 16
#     tiles. 64 tiles give 4 blocks, more than the 2 cores, so the op takes the
#     plain matmul (MatmulMultiCoreProgramConfig, no multicast), as the captured
#     call does. On a grid with 4 or more cores the 4 blocks fit, and the op takes
#     the 1D multicast matmul instead; test_matmul.py covers that grid with the
#     captured width.
#   * mirrors 04 (all operands in DRAM): an output 8 or more times wider than tall
#     takes the 1D multicast matmul (mcast_in0) on any grid, as the captured call
#     does.
# Both calls run on any grid size, so no variant needs to skip.
# ---------------------------------------------------------------------------
"""Narrow-weight companion of the ``ttnn.matmul`` lm_head cases (finishes on a 2-core grid)."""

import copy

import pytest

import ttnn
from models.experimental.ops.quasar.tests.gpt_oss_ops import graph_case as G
from models.experimental.ops.quasar.tests.gpt_oss_ops import test_matmul as T

_OP = ttnn.matmul

_W = 2048  # 64 tiles = 4 output blocks of 16 tiles

_LM_HEAD_IDS = ("03_1x2880_bf8_int-l1", "04_32x2880_bf8_int-dram")


def _narrow(case):
    """The captured case with the weight (and output) width set to _W."""
    small = copy.deepcopy(case)
    small["id"] = "small_" + case["id"].replace("x2880", f"x2880x{_W}", 1)
    small["count"] = 1
    weight = small["args"][1]
    assert weight["shape"][-1] == 262144, weight["shape"]
    weight["shape"][-1] = _W
    for out in small["outs"]:
        if out is not None:
            out["shape"][-1] = _W
    return small


CASES = [_narrow(c) for c in T.CASES if c["id"] in _LM_HEAD_IDS]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_matmul_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
