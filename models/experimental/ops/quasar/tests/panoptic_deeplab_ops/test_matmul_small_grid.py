# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_matmul.py — do NOT regenerate this file from a capture.
#
# The suite's \`\`-m emulator\`\` subset caps the activation at 48 x 48 x 256 elements, the
# total input footprint at 16 MiB and an L1 shard grid at 8 cores. The captured Panoptic-DeepLab
# call is at 512 x 1024 resolution on a 20-core device, so it is outside that subset. The variants
# below keep the call and shrink the tensors; dtypes are bfloat16 directly (the captured
# bfloat8_b has no Quasar format), so no TTNN_GRAPH_OPS_BF8_AS_BF16 is needed.
#
# The captured calls are the ASPP / decoder batched matmuls ([1, 32, 512, 128] x [1, 32, 128, 1024]
# and [1, 32, 128, 256] x [1, 1, 256, 1024]) under a 1D multicast program config on a 5 x 4 grid.
# The variants keep both operand forms (batched weight in L1, broadcast weight in DRAM) and the 1D
# multicast config (mcast_in0, fuse_batch 0) on a 2 x 1 grid: batch 2, M = 64 (2 tiles), K = 128
# (4 tiles), N = 128 (4 tiles), so per_core_M = 2 and per_core_N = 2 on the two cores, and in0_block_w
# 2 as captured. Case 02 is the same product without a program config, so the op chooses its own.
# The torch golden (matmul) applies unchanged.
# ---------------------------------------------------------------------------
"""Small (2-node) variants of ``ttnn.matmul``: the captured batched matmuls on a 2 x 1 grid."""

import copy

import pytest

import ttnn
from models.experimental.ops.quasar.tests.panoptic_deeplab_ops import graph_case as G

_OP = ttnn.matmul

_DRAM = {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None}
_L1 = {"layout": "INTERLEAVED", "buffer": "L1", "shard": None}

_M, _K, _N = 64, 128, 128

_PROGRAM_CONFIG = {
    "kind": "MatmulMultiCoreReuseMultiCast1DProgramConfig",
    "fields": {
        "compute_with_storage_grid_size": [2, 1],
        "in0_block_w": 2,
        "out_subblock_h": 1,
        "out_subblock_w": 2,
        "out_block_h": 2,
        "out_block_w": 2,
        "per_core_M": 2,
        "per_core_N": 2,
        "fuse_batch": 0,
        "fused_activation": None,
        "mcast_in0": 1,
        "gather_in0": 0,
        "hop_cores": {},
        "num_global_cb_receivers": 0,
        "untilize_out": 0,
        "allowed_worker_cores": None,
        "stream_in1": 0,
    },
    "k": "cfg",
}


def _case(case_id, batch_b, mem_a, mem_b, mem_out, program_config):
    kwargs = {"memory_config": {**mem_out, "k": "mem"}}
    if program_config:
        kwargs["program_config"] = copy.deepcopy(program_config)
    return {
        "id": case_id,
        "op": "ttnn.matmul",
        "count": 1,
        "args": [
            {"k": "t", "shape": [1, 2, _M, _K], "dtype": "BFLOAT16", "layout": "TILE", "mem": dict(mem_a)},
            {"k": "t", "shape": [1, batch_b, _K, _N], "dtype": "BFLOAT16", "layout": "TILE", "mem": dict(mem_b)},
        ],
        "kwargs": kwargs,
        "outs": [
            {"dtype": "BFLOAT16", "k": "t", "layout": "TILE", "mem": dict(mem_out), "shape": [1, 2, _M, _N]},
        ],
    }


CASES = [
    _case("00_2x64x128_bf16_int-dram", 2, _DRAM, _L1, _DRAM, _PROGRAM_CONFIG),
    _case("01_2x64x128_bf16_bcast_int-l1", 1, _L1, _DRAM, _L1, _PROGRAM_CONFIG),
    _case("02_2x64x128_bf16_default_int-dram", 2, _DRAM, _DRAM, _DRAM, None),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_matmul_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
