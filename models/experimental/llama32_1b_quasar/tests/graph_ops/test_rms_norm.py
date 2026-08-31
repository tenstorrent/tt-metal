# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# GENERATED FILE - do not edit by hand.
# Regenerate with:
#   python models/experimental/llama32_1b_quasar/tests/graph_ops/generate_from_graph_capture.py \
#       --capture generated/ttnn/reports/llama32_1b_demo_aug20_0223/graph_capture.python_io.json --out models/experimental/llama32_1b_quasar/tests/graph_ops
# Source capture: generated/ttnn/reports/llama32_1b_demo_aug20_0223/graph_capture.python_io.json
# ---------------------------------------------------------------------------
"""
Per-op test: ``ttnn.rms_norm`` — every distinct call the model made, as captured.

Captured 700 call(s) to this op; 4 distinct signature(s) covering 700 of them.

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
from models.experimental.llama32_1b_quasar.tests.graph_ops import graph_case as G

_OP = ttnn.rms_norm

CASES = [
    {
        "id": "00_32x2048_bf16_ws-l1",
        "op": "ttnn.rms_norm",
        "count": 327,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 2048],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 7, 3]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "epsilon": {"k": "lit", "v": 1e-05},
            "weight": {
                "k": "t",
                "shape": [1, 1, 64, 32],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "program_config": {
                "kind": "LayerNormShardedMultiCoreProgramConfig",
                "fields": {
                    "compute_with_storage_grid_size": [8, 4],
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
                "shard": {"grid": [[0, 0, 7, 3]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
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
                    "shard": {"grid": [[0, 0, 7, 3]], "orientation": "ROW_MAJOR", "shape": [32, 64]},
                },
                "shape": [1, 1, 32, 2048],
            },
        ],
    },
    {
        "id": "01_32x2048_bf16_ws-l1",
        "op": "ttnn.rms_norm",
        "count": 307,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 2048],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "WIDTH_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 7, 7]], "shape": [32, 32], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "epsilon": {"k": "lit", "v": 1e-05},
            "weight": {
                "k": "t",
                "shape": [1, 1, 64, 32],
                "dtype": "BFLOAT16",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
            "program_config": {
                "kind": "LayerNormShardedMultiCoreProgramConfig",
                "fields": {
                    "compute_with_storage_grid_size": [8, 8],
                    "subblock_w": 1,
                    "block_h": 1,
                    "block_w": 1,
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
                "shard": {"grid": [[0, 0, 7, 7]], "shape": [32, 32], "orientation": "ROW_MAJOR"},
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
                    "shard": {"grid": [[0, 0, 7, 7]], "orientation": "ROW_MAJOR", "shape": [32, 32]},
                },
                "shape": [1, 1, 32, 2048],
            },
        ],
    },
    {
        "id": "02_1024x2048_bf16_int-dram",
        "op": "ttnn.rms_norm",
        "count": 64,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1024, 2048],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "epsilon": {"k": "lit", "v": 1e-05},
            "weight": {
                "k": "t",
                "shape": [1, 1, 64, 32],
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
                "shape": [1, 1, 1024, 2048],
            },
        ],
    },
    {
        "id": "03_32x2048_bf16_int-dram",
        "op": "ttnn.rms_norm",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 2048],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {
            "epsilon": {"k": "lit", "v": 1e-05},
            "weight": {
                "k": "t",
                "shape": [1, 1, 64, 32],
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
                "shape": [1, 1, 32, 2048],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_rms_norm(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
