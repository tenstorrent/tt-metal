# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# GENERATED FILE - do not edit by hand.
# Regenerate with:
#   python models/experimental/llama32_1b_quasar/tests/graph_ops/generate_from_graph_capture.py \
#       --capture generated/ttnn/reports/gpt_oss_demo_aug28_1452/graph_capture.python_io.slim.json --out models/experimental/ops/quasar/tests/gpt_oss_ops
# Source capture: generated/ttnn/reports/gpt_oss_demo_aug28_1452/graph_capture.python_io.slim.json
# ---------------------------------------------------------------------------
"""
Per-op test: ``ttnn.sparse_matmul`` — every distinct call the model made, as captured.

Captured 720 call(s) to this op; 4 distinct signature(s) covering 720 of them.

Each CASES entry is one distinct call: the exact input shapes / dtypes / layouts /
memory configs, the keyword arguments (memory_config, program_config, scalars) and
one captured output spec per tensor the op returned. ``count`` is how many times
that exact call occurred in the captured run. See ``graph_case.py`` for how a case
is materialized and checked, and README.md for the fidelity caveats (random inputs,
no compute_kernel_config).
"""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.gpt_oss_ops import graph_case as G

_OP = ttnn.sparse_matmul

CASES = [
    {
        "id": "00_1x2880_bf8_int-l1",
        "op": "ttnn.sparse_matmul",
        "count": 384,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 32, 2880, 2880],
                "dtype": "BFLOAT4_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "sparsity": {
                "k": "t",
                "shape": [1, 1, 1, 32],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            "nnz": {"k": "lit", "v": None},
            "memory_config": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None, "k": "mem"},
            "output_tile": {"k": "tile", "shape": [32, 32]},
            "program_config": {
                "kind": "MatmulMultiCoreReuseMultiCast1DProgramConfig",
                "fields": {
                    "compute_with_storage_grid_size": [3, 4],
                    "in0_block_w": 30,
                    "out_subblock_h": 1,
                    "out_subblock_w": 1,
                    "out_block_h": 1,
                    "out_block_w": 1,
                    "per_core_M": 1,
                    "per_core_N": 8,
                    "fuse_batch": 0,
                    "fused_activation": None,
                    "mcast_in0": 1,
                    "gather_in0": 0,
                    "hop_cores": {},
                    "num_global_cb_receivers": 1,
                    "untilize_out": 0,
                    "allowed_worker_cores": None,
                    "stream_in1": 0,
                },
                "k": "cfg",
            },
            "dtype": {"k": "dtype", "v": "BFLOAT8_B"},
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1, 32, 1, 2880],
            },
        ],
    },
    {
        "id": "01_32x1x2880_bf8_int-l1",
        "op": "ttnn.sparse_matmul",
        "count": 192,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 1, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 32, 2880, 2880],
                "dtype": "BFLOAT4_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "sparsity": {
                "k": "t",
                "shape": [1, 1, 1, 32],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
            "nnz": {"k": "lit", "v": None},
            "memory_config": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None, "k": "mem"},
            "output_tile": {"k": "tile", "shape": [32, 32]},
            "is_input_a_sparse": {"k": "lit", "v": True},
            "program_config": {
                "kind": "MatmulMultiCoreReuseMultiCast1DProgramConfig",
                "fields": {
                    "compute_with_storage_grid_size": [5, 6],
                    "in0_block_w": 10,
                    "out_subblock_h": 1,
                    "out_subblock_w": 1,
                    "out_block_h": 1,
                    "out_block_w": 1,
                    "per_core_M": 1,
                    "per_core_N": 3,
                    "fuse_batch": 0,
                    "fused_activation": None,
                    "mcast_in0": 1,
                    "gather_in0": 0,
                    "hop_cores": {},
                    "num_global_cb_receivers": 1,
                    "untilize_out": 0,
                    "allowed_worker_cores": None,
                    "stream_in1": 0,
                },
                "k": "cfg",
            },
            "dtype": {"k": "dtype", "v": "BFLOAT8_B"},
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 1, 2880],
            },
        ],
    },
    {
        "id": "02_4x32x2880_bf16_int-dram",
        "op": "ttnn.sparse_matmul",
        "count": 96,
        "args": [
            {
                "k": "t",
                "shape": [1, 4, 32, 2880],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 32, 2880, 2880],
                "dtype": "BFLOAT4_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "sparsity": {
                "k": "t",
                "shape": [1, 1, 4, 32],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "nnz": {"k": "lit", "v": 128},
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
            "output_tile": {"k": "tile", "shape": [32, 32]},
            "program_config": {
                "kind": "MatmulMultiCoreReuseMultiCast1DProgramConfig",
                "fields": {
                    "compute_with_storage_grid_size": [3, 4],
                    "in0_block_w": 30,
                    "out_subblock_h": 1,
                    "out_subblock_w": 1,
                    "out_block_h": 1,
                    "out_block_w": 1,
                    "per_core_M": 1,
                    "per_core_N": 8,
                    "fuse_batch": 0,
                    "fused_activation": None,
                    "mcast_in0": 1,
                    "gather_in0": 0,
                    "hop_cores": {},
                    "num_global_cb_receivers": 1,
                    "untilize_out": 0,
                    "allowed_worker_cores": None,
                    "stream_in1": 0,
                },
                "k": "cfg",
            },
            "dtype": {"k": "dtype", "v": "BFLOAT8_B"},
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 4, 1, 32, 32, 2880],
            },
        ],
    },
    {
        "id": "03_32x128x2880_bf8_int-dram",
        "op": "ttnn.sparse_matmul",
        "count": 48,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 128, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            {
                "k": "t",
                "shape": [1, 32, 2880, 2880],
                "dtype": "BFLOAT4_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "sparsity": {
                "k": "t",
                "shape": [1, 1, 1, 32],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "nnz": {"k": "lit", "v": 32},
            "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
            "output_tile": {"k": "tile", "shape": [32, 32]},
            "is_input_a_sparse": {"k": "lit", "v": True},
            "program_config": {
                "kind": "MatmulMultiCoreReuseMultiCast1DProgramConfig",
                "fields": {
                    "compute_with_storage_grid_size": [5, 6],
                    "in0_block_w": 10,
                    "out_subblock_h": 1,
                    "out_subblock_w": 1,
                    "out_block_h": 1,
                    "out_block_w": 1,
                    "per_core_M": 4,
                    "per_core_N": 3,
                    "fuse_batch": 0,
                    "fused_activation": None,
                    "mcast_in0": 1,
                    "gather_in0": 0,
                    "hop_cores": {},
                    "num_global_cb_receivers": 1,
                    "untilize_out": 0,
                    "allowed_worker_cores": None,
                    "stream_in1": 0,
                },
                "k": "cfg",
            },
            "dtype": {"k": "dtype", "v": "BFLOAT8_B"},
        },
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 128, 2880],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_sparse_matmul(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
