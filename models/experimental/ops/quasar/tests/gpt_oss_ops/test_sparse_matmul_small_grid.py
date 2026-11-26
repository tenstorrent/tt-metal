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
# The shapes are reduced as well, so the variants fit the 2-node emulator and its
# ``-m emulator`` subset (the captured calls carry 32 experts x 2880 x 2880 weights, 142 MiB or
# more, and the prefill case an activation of 4096 rows). Each variant keeps the 32 experts
# (the sparsity tensor's 32-wide page and the expert dimension stay as captured) and cuts the
# hidden and intermediate widths from 2880 to 256, so a weight is [1, 32, 256, 256] bfloat16
# (4 MiB in DRAM). K = 256 is 8 tiles, so in0_block_w becomes 4 (it was 30 / 10 of 90 tiles) and
# the grid fitting below sets per_core_N from the 8 output tiles. The captured bfloat8_b /
# bfloat4_b dtypes become bfloat16 directly. Case 03 (the 4096-row prefill) has no reduced
# form: with 32 experts it needs at least 32 x 32 = 1024 activation rows, above the emulator
# subset's 128-row cap, so the decode (00, 01) and the 4-token (02) cases are covered.
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

_OP = ttnn.sparse_matmul

_DRAM = {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None}
_L1 = {"layout": "INTERLEAVED", "buffer": "L1", "shard": None}

_EXPERTS = 32
_WIDTH = 256  # replaces the captured 2880 (hidden and intermediate)
_IN0_BLOCK_W = 4  # divides K = 256 / 32 = 8 tiles
_WEIGHT_DTYPE = "BFLOAT16"

# Case ids of the decode down projection, whose output placement follows the model's rule (see the header).
_DECODE_DOWN_IDS = {"01_32x1x256_bf16_int-l1"}


def _case(case_id, a_shape, out_shape, sparsity_shape, mem, nnz, is_input_a_sparse, grid):
    fields = {
        "compute_with_storage_grid_size": list(grid),
        "in0_block_w": _IN0_BLOCK_W,
        "out_subblock_h": 1,
        "out_subblock_w": 1,
        "out_block_h": 1,
        "out_block_w": 1,
        "per_core_M": 1,
        "per_core_N": 1,
        "fuse_batch": 0,
        "fused_activation": None,
        "mcast_in0": 1,
        "gather_in0": 0,
        "hop_cores": {},
        "num_global_cb_receivers": 1,
        "untilize_out": 0,
        "allowed_worker_cores": None,
        "stream_in1": 0,
    }
    kwargs = {
        "sparsity": {"k": "t", "shape": sparsity_shape, "dtype": "BFLOAT16", "layout": "ROW_MAJOR", "mem": dict(mem)},
        "nnz": {"k": "lit", "v": nnz},
        "memory_config": {**mem, "k": "mem"},
        "output_tile": {"k": "tile", "shape": [32, 32]},
    }
    if is_input_a_sparse:
        kwargs["is_input_a_sparse"] = {"k": "lit", "v": True}
    kwargs["program_config"] = {
        "kind": "MatmulMultiCoreReuseMultiCast1DProgramConfig",
        "fields": fields,
        "k": "cfg",
    }
    kwargs["dtype"] = {"k": "dtype", "v": _WEIGHT_DTYPE}
    return {
        "id": case_id,
        "op": "ttnn.sparse_matmul",
        "count": 1,
        "args": [
            {"k": "t", "shape": a_shape, "dtype": _WEIGHT_DTYPE, "layout": "TILE", "mem": dict(mem)},
            {
                "k": "t",
                "shape": [1, _EXPERTS, _WIDTH, _WIDTH],
                "dtype": _WEIGHT_DTYPE,
                "layout": "TILE",
                "mem": dict(_DRAM),
            },
        ],
        "kwargs": kwargs,
        "outs": [{"dtype": _WEIGHT_DTYPE, "k": "t", "layout": "TILE", "mem": dict(mem), "shape": out_shape}],
    }


# The captured grids ((3, 4) gate/up, (5, 6) down) would leave cores idle on 8 output tiles, so the reduced cases
# start from (2, 4) (8 cores, one tile each); _fit_case then applies the model's rule for the device.
_GRID = (2, 4)
_REDUCED = [
    # 00: decode gate/up, one token against every expert
    _case(
        "00_1x256_bf16_int-l1",
        [1, 1, 1, _WIDTH],
        [1, 1, 1, _EXPERTS, 1, _WIDTH],
        [1, 1, 1, _EXPERTS],
        _L1,
        None,
        False,
        _GRID,
    ),
    # 01: decode down projection, per-expert token, input A sparse
    _case(
        "01_32x1x256_bf16_int-l1",
        [1, _EXPERTS, 1, _WIDTH],
        [1, _EXPERTS, 1, _WIDTH],
        [1, 1, 1, _EXPERTS],
        _L1,
        None,
        True,
        _GRID,
    ),
    # 02: 4-token gate/up
    _case(
        "02_4x32x256_bf16_int-dram",
        [1, 4, 32, _WIDTH],
        [1, 4, 1, _EXPERTS, 32, _WIDTH],
        [1, 1, 4, _EXPERTS],
        _DRAM,
        128,
        False,
        _GRID,
    ),
]

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


CASES = [pytest.param(c, id=f"{c['id']}-fit_grid") for c in _REDUCED]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES)
def test_sparse_matmul_small_grid(ttnn_mesh_device, reset_seeds, case):
    fitted = _fit_case(case, ttnn_mesh_device)
    G.run_case(_OP, fitted, ttnn_mesh_device)
