# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_sparse_matmul.py — do NOT regenerate this
# file from a capture.
#
# The captured cases are the gpt-oss expert projections. Their program configs
# (MatmulMultiCoreReuseMultiCast1DProgramConfig) set compute_with_storage_grid_size
# to the model's default expert grids: (3, 4) for the gate/up projections (cases 00
# and 02) and (5, 6) for the down projections (cases 01 and 03). On a device with a
# smaller compute grid (for example the 2x1 grid of the 2-core Quasar emulator) the
# captured grids do not exist, and the op fails with "Multicast kernel is placed
# outside the worker grid" (3, 4) or "No core coordinate found at location" (5, 6).
#
# The gpt-oss model does not use these grids as they are: mlp.py builds its expert
# program configs with max_grid = get_compute_grid_size(device), and
# ProgramConfig._build_matmul_config (gpt_oss/tt/experts/config.py) replaces a grid
# that does not fit with fit_core_grid() and sets per_core_N = ceil(Nt / num_cores).
# Each variant here applies the same rule to the captured case at run time, with the
# grid of the device under test. On a 2x1 device
# every projection gets (2, 1) and per_core_N = 45 (Nt = 2880 / 32 = 90). All other
# program config fields (in0_block_w, subblocks, mcast_in0, ...) stay as captured,
# because the model computes them without the grid.
#
# Case 01 (decode down projection) can have one more change. Its input A and its output
# are both [1, 32, 1, 2880], which is 32 tile rows of 90 tiles each. As bfloat16 (the
# suite runs the captured bfloat8_b as bfloat16, because Quasar has no block-float
# formats) each tensor is 5.9 MB, which does not fit in the SRAM of 2 cores next to
# input A. The model places this output with down_output_memory_config
# (gpt_oss/tt/experts/decode.py): in SRAM as captured where it fits, in interleaved DRAM
# where it does not. The variant uses the same rule and keeps input A in SRAM as captured.
#
# On a device on which a case needs no change (cases 00 and 02 on the 8x4 grid, where
# (3, 4) fits), the variant skips, because the generated test runs that exact call.
#
# The suite has no golden reference for sparse_matmul (see GOLDEN in graph_case.py),
# so the checks are the same as in the generated test: output shape, dtype, placement
# and finite values.
# ---------------------------------------------------------------------------
"""Small-grid variants of ``ttnn.sparse_matmul``: the captured cases with the model's expert grid fitting."""

import copy
import math

import pytest

import torch

import ttnn
from models.experimental.ops.quasar.gpt_oss.tt.experts.config import fit_core_grid
from models.experimental.ops.quasar.gpt_oss.tt.experts.decode import down_output_memory_config
from models.experimental.ops.quasar.gpt_oss.utils.general_utils import get_compute_grid_size
from models.experimental.ops.quasar.tests.gpt_oss_ops import graph_case as G
from models.experimental.ops.quasar.tests.gpt_oss_ops.test_sparse_matmul import CASES as _CAPTURED

_OP = ttnn.sparse_matmul

# Captured case ids of the decode down projection, whose output placement follows the model's rule (see the header).
_DECODE_DOWN_IDS = {"01_32x1x2880_bf8_int-l1"}

_DRAM_MEM = {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None}


def _output_in_dram(case, mesh_device):
    """Whether the model's down_output_memory_config puts the output of this case in DRAM on this device."""
    in_spec = case["args"][0]
    down_input = ttnn.from_torch(
        torch.zeros(in_spec["shape"]), dtype=G.DTYPE[in_spec["dtype"]], layout=G.LAYOUT[in_spec["layout"]]
    )
    output_width = case["args"][1]["shape"][-1]
    return down_output_memory_config(down_input, output_width, mesh_device) == ttnn.DRAM_MEMORY_CONFIG


def _fit_case(base, mesh_device):
    """The captured case with the program config the model would build for this device."""
    case = copy.deepcopy(base)
    fields = case["kwargs"]["program_config"]["fields"]
    grid = get_compute_grid_size(mesh_device)
    num_tiles_n = math.ceil(case["args"][1]["shape"][-1] / 32)
    core_x, core_y = fit_core_grid(tuple(fields["compute_with_storage_grid_size"]), num_tiles_n, (grid.x, grid.y))
    fields["compute_with_storage_grid_size"] = [core_x, core_y]
    fields["per_core_N"] = math.ceil(num_tiles_n / (core_x * core_y))
    if base["id"] in _DECODE_DOWN_IDS and _output_in_dram(case, mesh_device):
        case["kwargs"]["memory_config"] = {**_DRAM_MEM, "k": "mem"}
        case["outs"][0]["mem"] = dict(_DRAM_MEM)
    return case


CASES = [pytest.param(c, id=f"{c['id']}-fit_grid") for c in _CAPTURED]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES)
def test_sparse_matmul_small_grid(ttnn_mesh_device, reset_seeds, case):
    fitted = _fit_case(case, ttnn_mesh_device)
    if fitted == case:
        pytest.skip("the captured call needs no change on this device; test_sparse_matmul.py runs it")
    G.run_case(_OP, fitted, ttnn_mesh_device)
