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
Per-op test: ``ttnn.rms_norm`` — every distinct call the model made, as captured.

Captured 58290 call(s) to this op; 8 distinct signature(s) covering 58290 of them.

Fidelity notes for this op:
  * compute_kernel_config is recorded only as an object address in the capture, so it is dropped and the op's default is used (can shift PCC, not shapes)

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

_OP = ttnn.rms_norm

CASES = [
    {
        "id": "00_32x2560_bf16_ws-l1",
        "op": "ttnn.rms_norm",
        "count": 14800,
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
        ],
        "kwargs": {
            "epsilon": {"k": "lit", "v": 1e-06},
            "weight": {
                "k": "t",
                "shape": [1, 1, 80, 32],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "program_config": {
                "kind": "LayerNormShardedMultiCoreProgramConfig",
                "fields": {
                    "compute_with_storage_grid_size": [8, 5],
                    "subblock_w": 2,
                    "block_h": 1,
                    "block_w": 2,
                    "inplace": 0,
                    "legacy_reduction": 0,
                    "legacy_rsqrt": 0,
                    "use_welford": 0,
                },
                "k": "cfg",
            },
            "memory_config": {
                "layout": "WIDTH_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 7, 4]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                "k": "mem",
            },
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {
                    "buffer": "L1",
                    "layout": "WIDTH_SHARDED",
                    "shard": {"grid": [[0, 0, 7, 4]], "orientation": "ROW_MAJOR", "shape": [32, 64]},
                },
                "shape": [1, 1, 32, 2560],
            },
        ],
    },
    {
        "id": "01_32x128_bf16_int-l1",
        "op": "ttnn.rms_norm",
        "count": 14400,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 128],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
        ],
        "kwargs": {
            "epsilon": {"k": "lit", "v": 1e-06},
            "weight": {
                "k": "t",
                "shape": [1, 1, 4, 32],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "program_config": {"k": "lit", "v": None},
            "memory_config": {"k": "lit", "v": None},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 32, 128],
            },
        ],
    },
    {
        "id": "02_8x128_bf16_int-l1",
        "op": "ttnn.rms_norm",
        "count": 14400,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 8, 128],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
        ],
        "kwargs": {
            "epsilon": {"k": "lit", "v": 1e-06},
            "weight": {
                "k": "t",
                "shape": [1, 1, 4, 32],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "program_config": {"k": "lit", "v": None},
            "memory_config": {"k": "lit", "v": None},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 8, 128],
            },
        ],
    },
    {
        "id": "03_32x2560_bf16_ws-l1",
        "op": "ttnn.rms_norm",
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
                    "shard": {"grid": [[0, 0, 7, 1]], "shape": [32, 160], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "epsilon": {"k": "lit", "v": 1e-06},
            "weight": {
                "k": "t",
                "shape": [1, 1, 80, 32],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "program_config": {
                "kind": "LayerNormShardedMultiCoreProgramConfig",
                "fields": {
                    "compute_with_storage_grid_size": [8, 2],
                    "subblock_w": 1,
                    "block_h": 1,
                    "block_w": 5,
                    "inplace": 0,
                    "legacy_reduction": 0,
                    "legacy_rsqrt": 0,
                    "use_welford": 0,
                },
                "k": "cfg",
            },
            "memory_config": {
                "layout": "WIDTH_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 7, 1]], "shape": [32, 160], "orientation": "ROW_MAJOR"},
                "k": "mem",
            },
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
        "id": "04_4096x2560_bf16_int-dram",
        "op": "ttnn.rms_norm",
        "count": 144,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 4096, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "epsilon": {"k": "lit", "v": 1e-06},
            "weight": {
                "k": "t",
                "shape": [1, 1, 80, 32],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "program_config": {"k": "lit", "v": None},
            "memory_config": {"k": "lit", "v": None},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 4096, 2560],
            },
        ],
    },
    {
        "id": "05_32x4096x128_bf16_int-dram",
        "op": "ttnn.rms_norm",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 4096, 128],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "epsilon": {"k": "lit", "v": 1e-06},
            "weight": {
                "k": "t",
                "shape": [1, 1, 4, 32],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "program_config": {"k": "lit", "v": None},
            "memory_config": {"k": "lit", "v": None},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 32, 4096, 128],
            },
        ],
    },
    {
        "id": "06_8x4096x128_bf16_int-dram",
        "op": "ttnn.rms_norm",
        "count": 72,
        "args": [
            {
                "k": "t",
                "shape": [1, 8, 4096, 128],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "epsilon": {"k": "lit", "v": 1e-06},
            "weight": {
                "k": "t",
                "shape": [1, 1, 4, 32],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "program_config": {"k": "lit", "v": None},
            "memory_config": {"k": "lit", "v": None},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 8, 4096, 128],
            },
        ],
    },
    {
        "id": "07_32x2560_bf16_int-dram",
        "op": "ttnn.rms_norm",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 2560],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "epsilon": {"k": "lit", "v": 1e-06},
            "weight": {
                "k": "t",
                "shape": [1, 1, 80, 32],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "program_config": {"k": "lit", "v": None},
            "memory_config": {"k": "lit", "v": None},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 32, 2560],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_rms_norm(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
