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
# grid of the device under test and the grid that mlp.py excludes. On a 2x1 device
# every projection gets (2, 1) and per_core_N = 45 (Nt = 2880 / 32 = 90). All other
# program config fields (in0_block_w, subblocks, mcast_in0, ...) stay as captured,
# because the model computes them without the grid.
#
# Case 01 (decode down projection) has one more change. Its input A and its output are
# both [1, 32, 1, 2880] in L1, which is 32 tile rows of 90 tiles each. As bfloat16 (the
# suite runs the captured bfloat8_b as bfloat16, because Quasar has no block-float
# formats) each tensor is 5.9 MB. On 2 cores the SRAM is split into 2 banks of 3.86 MB:
# input A takes 2.95 MB of each bank, so the output (another 2.95 MB per bank) does not
# fit ("Out of Memory: Not enough space to allocate 5898240 B L1 buffer across 2
# banks"). The variant keeps input A in L1 as captured and moves the output to
# interleaved DRAM, the placement the model's prefill down projection uses
# (gpt_oss/tt/experts/prefill.py). The circular buffers of this op are a few blocks of
# in0_block_w tiles each, so they fit next to input A.
#
# On a device on which a case needs no change (cases 00 and 02 on the 8x4 grid, where
# (3, 4) fits), the variant skips, because the generated test runs that exact call.
#
# On the (2, 1) and (1, 1) grids, every case stops the craq-sim simulator, which ends the
# whole pytest process: "UndefinedBehavior: qsr_tile_counter_check_error: tile counter
# occupancy=... exceeds capacity=... (posted=0 acked=...)". This is the error that made
# mlp.py exclude (6, 3). It does not depend on TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE:
# case 01 forced to (2, 1) on the full 8x4 grid fails the same way, while (3, 1) passes.
# A 2-core device has no other grid, so on these grids the variant is marked xfail
# without running.
#
# The suite has no golden reference for sparse_matmul (see GOLDEN in graph_case.py),
# so the checks are the same as in the generated test: output shape, dtype, placement
# and finite values.
# ---------------------------------------------------------------------------
"""Small-grid variants of ``ttnn.sparse_matmul``: the captured cases with the model's expert grid fitting."""

import copy
import math

import pytest

import ttnn
from models.experimental.ops.quasar.gpt_oss.tt.experts.config import fit_core_grid
from models.experimental.ops.quasar.gpt_oss.utils.general_utils import get_compute_grid_size
from models.experimental.ops.quasar.tests.gpt_oss_ops import graph_case as G
from models.experimental.ops.quasar.tests.gpt_oss_ops.test_sparse_matmul import CASES as _CAPTURED

_OP = ttnn.sparse_matmul

# The grids that mlp.py passes as excluded_grids to GPTOSSProgramConfig.
_EXCLUDED_GRIDS = ((6, 3),)

# Captured case ids whose output moves from L1 to DRAM (see the header).
_DRAM_OUTPUT_IDS = {"01_32x1x2880_bf8_int-l1"}

_DRAM_MEM = {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None}

# Grids on which every case stops the craq-sim simulator with a tile counter error (see the header).
_STOPS_SIMULATOR_GRIDS = {(1, 1), (2, 1)}


def _fit_case(base, mesh_device):
    """The captured case with the program config the model would build for this device."""
    case = copy.deepcopy(base)
    fields = case["kwargs"]["program_config"]["fields"]
    grid = get_compute_grid_size(mesh_device)
    num_tiles_n = math.ceil(case["args"][1]["shape"][-1] / 32)
    core_x, core_y = fit_core_grid(
        tuple(fields["compute_with_storage_grid_size"]), num_tiles_n, (grid.x, grid.y), _EXCLUDED_GRIDS
    )
    fields["compute_with_storage_grid_size"] = [core_x, core_y]
    fields["per_core_N"] = math.ceil(num_tiles_n / (core_x * core_y))
    if base["id"] in _DRAM_OUTPUT_IDS:
        case["kwargs"]["memory_config"] = {**_DRAM_MEM, "k": "mem"}
        case["outs"][0]["mem"] = dict(_DRAM_MEM)
    return case


CASES = [
    pytest.param(c, id=f"{c['id']}-fit_grid" + ("-dram_out" if c["id"] in _DRAM_OUTPUT_IDS else "")) for c in _CAPTURED
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES)
def test_sparse_matmul_small_grid(ttnn_mesh_device, reset_seeds, case):
    fitted = _fit_case(case, ttnn_mesh_device)
    if fitted == case:
        pytest.skip("the captured call needs no change on this device; test_sparse_matmul.py runs it")
    grid = tuple(fitted["kwargs"]["program_config"]["fields"]["compute_with_storage_grid_size"])
    if grid in _STOPS_SIMULATOR_GRIDS:
        # Not run: the simulator error ends the pytest process, so a failing run cannot be caught.
        pytest.xfail(f"sparse_matmul on a {grid} grid stops craq-sim with a tile counter error")
    G.run_case(_OP, fitted, ttnn_mesh_device)
