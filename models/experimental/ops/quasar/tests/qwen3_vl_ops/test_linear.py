# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# GENERATED FILE - do not edit by hand.
# Regenerate with:
#   python models/experimental/llama32_1b_quasar/tests/graph_ops/generate_from_graph_capture.py \
#       --capture generated/ttnn/reports/qwen3_vl_demo_aug27_1509/graph_capture.python_io.slim.json --out models/experimental/ops/quasar/tests/qwen3_vl_ops
# Source capture: generated/ttnn/reports/qwen3_vl_demo_aug27_1509/graph_capture.python_io.slim.json
# ---------------------------------------------------------------------------
"""
Per-op test: ``ttnn.linear`` — every distinct call the model made, as captured.

Captured 74940 call(s) to this op; 12 distinct signature(s) covering 74856 of them.

Fidelity notes for this op:
  * compute_kernel_config is recorded only as an object address in the capture, so it is dropped and the op's default is used (can shift PCC, not shapes)
  * the output has more elements than all inputs combined (a batch-padded decode tensor), so the op cannot write all of it; finiteness is asserted over the portion the inputs can account for

NOT generated (arguments not reconstructible from the capture):
  * 72 call(s): argument(s) activation
  * 12 call(s): argument(s) activation

Each CASES entry is one distinct call: the exact input shapes / dtypes / layouts /
memory configs, the keyword arguments (memory_config, program_config, scalars) and
one captured output spec per tensor the op returned. ``count`` is how many times
that exact call occurred in the captured run. See ``graph_case.py`` for how a case
is materialized and checked, and README.md for the fidelity caveats (random inputs,
no compute_kernel_config).
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G

_OP = ttnn.linear

CASES = [
    {
        "id": "00_32x2560_bf16_ws-l1",
        "op": "ttnn.linear",
        "count": 28800,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 7, 1]], "shape": [32, 160], "orientation": "ROW_MAJOR"},
                },
            },
            {
                "k": "t",
                "shape": [1, 1, 2560, 9728],
                "dtype": "BFLOAT4_B",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "DRAM",
                    "shard": {"grid": [[0, 0, 11, 0]], "shape": [2560, 832], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "dtype": {"k": "dtype", "v": "BFLOAT16"},
            "core_grid": {"k": "lit", "v": None},
            "program_config": {
                "kind": "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig",
                "fields": {"in0_block_w": 5, "per_core_M": 1, "per_core_N": 19, "fused_activation": None},
                "k": "cfg",
            },
            "memory_config": {"layout": "WIDTH_SHARDED", "buffer": "L1", "shard": None, "k": "mem"},
            "global_cb": {"k": "lit", "v": None},
            "sub_device_id": {"k": "lit", "v": None},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "WIDTH_SHARDED",
                    "shard": {"grid": [[0, 0, 7, 1]], "orientation": "ROW_MAJOR", "shape": [32, 608]},
                },
                "shape": [1, 1, 32, 9728],
            },
        ],
    },
    {
        "id": "01_32x4096_bf16_ws-l1",
        "op": "ttnn.linear",
        "count": 14400,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 4096],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 7, 3]], "shape": [32, 128], "orientation": "ROW_MAJOR"},
                },
            },
            {
                "k": "t",
                "shape": [1, 1, 4096, 2560],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "DRAM",
                    "shard": {"grid": [[0, 0, 11, 0]], "shape": [4096, 224], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "core_grid": {"k": "lit", "v": None},
            "program_config": {
                "kind": "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig",
                "fields": {"in0_block_w": 4, "per_core_M": 1, "per_core_N": 3, "fused_activation": None},
                "k": "cfg",
            },
            "memory_config": {"layout": "WIDTH_SHARDED", "buffer": "L1", "shard": None, "k": "mem"},
            "dtype": {"k": "lit", "v": None},
            "global_cb": {"k": "lit", "v": None},
            "sub_device_id": {"k": "lit", "v": None},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "WIDTH_SHARDED",
                    "shard": {"grid": [[0, 0, 7, 2], [0, 3, 2, 3]], "orientation": "ROW_MAJOR", "shape": [32, 96]},
                },
                "shape": [1, 1, 32, 2560],
            },
        ],
    },
    {
        "id": "02_32x2560_bf16_ws-l1",
        "op": "ttnn.linear",
        "count": 14400,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 7, 4]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                },
            },
            {
                "k": "t",
                "shape": [1, 1, 2560, 6144],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "DRAM",
                    "shard": {"grid": [[0, 0, 11, 0]], "shape": [2560, 512], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "memory_config": {"layout": "WIDTH_SHARDED", "buffer": "L1", "shard": None, "k": "mem"},
            "program_config": {
                "kind": "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig",
                "fields": {"in0_block_w": 2, "per_core_M": 1, "per_core_N": 5, "fused_activation": None},
                "k": "cfg",
            },
            "dtype": {"k": "dtype", "v": "BFLOAT16"},
            "global_cb": {"k": "lit", "v": None},
            "sub_device_id": {"k": "lit", "v": None},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "WIDTH_SHARDED",
                    "shard": {"grid": [[0, 0, 7, 3], [0, 4, 6, 4]], "orientation": "ROW_MAJOR", "shape": [32, 160]},
                },
                "shape": [1, 1, 32, 6144],
            },
        ],
    },
    {
        "id": "03_32x9728_bf8_ws-l1",
        "op": "ttnn.linear",
        "count": 14400,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 9728],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 7, 1]], "shape": [32, 608], "orientation": "ROW_MAJOR"},
                },
            },
            {
                "k": "t",
                "shape": [1, 1, 9728, 2560],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "DRAM",
                    "shard": {"grid": [[0, 0, 11, 0]], "shape": [9728, 224], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "dtype": {"k": "dtype", "v": "BFLOAT16"},
            "program_config": {
                "kind": "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig",
                "fields": {"in0_block_w": 1, "per_core_M": 1, "per_core_N": 5, "fused_activation": None},
                "k": "cfg",
            },
            "memory_config": {"layout": "WIDTH_SHARDED", "buffer": "L1", "shard": None, "k": "mem"},
            "core_grid": {"k": "lit", "v": None},
            "global_cb": {"k": "lit", "v": None},
            "sub_device_id": {"k": "lit", "v": None},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "WIDTH_SHARDED",
                    "shard": {"grid": [[0, 0, 7, 1]], "orientation": "ROW_MAJOR", "shape": [32, 160]},
                },
                "shape": [1, 1, 32, 2560],
            },
        ],
    },
    {
        "id": "04_32x2560_bf16_ws-l1",
        "op": "ttnn.linear",
        "count": 2010,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 7, 4]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                },
            },
            {
                "k": "t",
                "shape": [2560, 26720],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "DRAM",
                    "shard": {"grid": [[0, 0, 11, 0]], "shape": [2560, 2240], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "program_config": {
                "kind": "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig",
                "fields": {"in0_block_w": 2, "per_core_M": 1, "per_core_N": 21, "fused_activation": None},
                "k": "cfg",
            },
            "memory_config": {"layout": "WIDTH_SHARDED", "buffer": "L1", "shard": None, "k": "mem"},
            "dtype": {"k": "dtype", "v": "BFLOAT8_B"},
            "sub_device_id": {"k": "lit", "v": None},
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "WIDTH_SHARDED",
                    "shard": {"grid": [[0, 0, 7, 4]], "orientation": "ROW_MAJOR", "shape": [32, 672]},
                },
                "shape": [1, 1, 32, 26720],
            },
        ],
    },
    {
        "id": "05_32x2560_bf16_ws-l1",
        "op": "ttnn.linear",
        "count": 402,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 7, 4]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                },
            },
            {
                "k": "t",
                "shape": [2560, 18336],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "DRAM",
                    "shard": {"grid": [[0, 0, 11, 0]], "shape": [2560, 1536], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "program_config": {
                "kind": "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig",
                "fields": {"in0_block_w": 2, "per_core_M": 1, "per_core_N": 15, "fused_activation": None},
                "k": "cfg",
            },
            "memory_config": {"layout": "WIDTH_SHARDED", "buffer": "L1", "shard": None, "k": "mem"},
            "dtype": {"k": "dtype", "v": "BFLOAT8_B"},
            "sub_device_id": {"k": "lit", "v": None},
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "WIDTH_SHARDED",
                    "shard": {"grid": [[0, 0, 7, 3], [0, 4, 6, 4]], "orientation": "ROW_MAJOR", "shape": [32, 480]},
                },
                "shape": [1, 1, 32, 18336],
            },
        ],
    },
    {
        "id": "06_4x1024x2560_bf16_int-dram",
        "op": "ttnn.linear",
        "count": 144,
        "args": [
            {
                "k": "t",
                "shape": [1, 4, 1024, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 1, 2560, 9728],
                "dtype": "BFLOAT4_B",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "DRAM",
                    "shard": {"grid": [[0, 0, 11, 0]], "shape": [2560, 832], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "dtype": {"k": "dtype", "v": "BFLOAT16"},
            "core_grid": {"k": "lit", "v": None},
            "program_config": {
                "kind": "MatmulMultiCoreReuseMultiCastProgramConfig",
                "fields": {
                    "compute_with_storage_grid_size": [8, 8],
                    "in0_block_w": 5,
                    "out_subblock_h": 1,
                    "out_subblock_w": 2,
                    "out_block_h": 4,
                    "out_block_w": 38,
                    "per_core_M": 4,
                    "per_core_N": 38,
                    "transpose_mcast": 0,
                    "fused_activation": None,
                    "fuse_batch": 0,
                    "allowed_worker_cores": None,
                },
                "k": "cfg",
            },
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
            "global_cb": {"k": "lit", "v": None},
            "sub_device_id": {"k": "lit", "v": None},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 4, 1024, 9728],
            },
        ],
    },
    {
        "id": "07_12x1024x4096_bf16_int-dram",
        "op": "ttnn.linear",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 12, 1024, 4096],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [4096, 1024],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "bias": {
                "k": "t",
                "shape": [1024],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 12, 1024, 1024],
            },
        ],
    },
    {
        "id": "08_6x2048x1024_bf16_int-dram",
        "op": "ttnn.linear",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 6, 2048, 1024],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 1, 1024, 3072],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "dtype": {"k": "lit", "v": None},
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
            "program_config": {
                "kind": "MatmulMultiCoreReuseMultiCastProgramConfig",
                "fields": {
                    "compute_with_storage_grid_size": [8, 8],
                    "in0_block_w": 1,
                    "out_subblock_h": 1,
                    "out_subblock_w": 1,
                    "out_block_h": 8,
                    "out_block_w": 12,
                    "per_core_M": 8,
                    "per_core_N": 12,
                    "transpose_mcast": 0,
                    "fused_activation": None,
                    "fuse_batch": 0,
                    "allowed_worker_cores": None,
                },
                "k": "cfg",
            },
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 6, 2048, 3072],
            },
        ],
    },
    {
        "id": "09_12x1024x1024_bf8_int-dram",
        "op": "ttnn.linear",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 12, 1024, 1024],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 1, 1024, 1024],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "dtype": {"k": "dtype", "v": "BFLOAT8_B"},
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
            "program_config": {
                "kind": "MatmulMultiCoreReuseMultiCastProgramConfig",
                "fields": {
                    "compute_with_storage_grid_size": [8, 8],
                    "in0_block_w": 1,
                    "out_subblock_h": 1,
                    "out_subblock_w": 4,
                    "out_block_h": 8,
                    "out_block_w": 4,
                    "per_core_M": 8,
                    "per_core_N": 4,
                    "transpose_mcast": 0,
                    "fused_activation": None,
                    "fuse_batch": 0,
                    "allowed_worker_cores": None,
                },
                "k": "cfg",
            },
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 12, 1024, 1024],
            },
        ],
    },
    {
        "id": "10_4x1024x4096_bf8_int-dram",
        "op": "ttnn.linear",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 4, 1024, 4096],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 1, 4096, 2560],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "DRAM",
                    "shard": {"grid": [[0, 0, 11, 0]], "shape": [4096, 224], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "dtype": {"k": "dtype", "v": "BFLOAT8_B"},
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
            "program_config": {
                "kind": "MatmulMultiCoreReuseMultiCastProgramConfig",
                "fields": {
                    "compute_with_storage_grid_size": [8, 8],
                    "in0_block_w": 8,
                    "out_subblock_h": 1,
                    "out_subblock_w": 2,
                    "out_block_h": 4,
                    "out_block_w": 10,
                    "per_core_M": 4,
                    "per_core_N": 10,
                    "transpose_mcast": 0,
                    "fused_activation": None,
                    "fuse_batch": 0,
                    "allowed_worker_cores": None,
                },
                "k": "cfg",
            },
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 4, 1024, 2560],
            },
        ],
    },
    {
        "id": "11_2752x4096_bf16_int-dram",
        "op": "ttnn.linear",
        "count": 12,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 2752, 4096],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [4096, 2560],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "bias": {
                "k": "t",
                "shape": [2560],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 2752, 2560],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_linear(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
