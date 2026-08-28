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
Per-op test: ``ttnn.unsqueeze_to_4D`` — every distinct call the model made, as captured.

Captured 698 call(s) to this op; 8 distinct signature(s) covering 698 of them.

Fidelity notes for this op:
  * an integer index tensor is involved; the capture holds no values, so it is filled by graph_case.INDEX_VALUES (page ids, positions, token ids) instead of random data

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

_OP = ttnn.unsqueeze_to_4D

CASES = [
    {
        "id": "00_1x32_bf16_int-l1",
        "op": "ttnn.unsqueeze_to_4D",
        "count": 192,
        "args": [
            {
                "k": "t",
                "shape": [1, 32],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1, 32],
            },
        ],
    },
    {
        "id": "01_1x2880_bf8_int-l1",
        "op": "ttnn.unsqueeze_to_4D",
        "count": 192,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 1, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1, 2880],
            },
        ],
    },
    {
        "id": "02_1x2880_bf8_int-l1",
        "op": "ttnn.unsqueeze_to_4D",
        "count": 192,
        "args": [
            {
                "k": "t",
                "shape": [1, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "L1", "shard": None},
            },
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "L1", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1, 2880],
            },
        ],
    },
    {
        "id": "03_128x2880_bf16_int-dram",
        "op": "ttnn.unsqueeze_to_4D",
        "count": 48,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 128, 2880],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 128, 2880],
            },
        ],
    },
    {
        "id": "04_128x2880_bf8_int-dram",
        "op": "ttnn.unsqueeze_to_4D",
        "count": 48,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 128, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 128, 2880],
            },
        ],
    },
    {
        "id": "05_32x64_bf16_int-dram",
        "op": "ttnn.unsqueeze_to_4D",
        "count": 16,
        "args": [
            {
                "k": "t",
                "shape": [1, 32, 64],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 32, 64],
            },
        ],
    },
    {
        "id": "06_32_u32_int-dram",
        "op": "ttnn.unsqueeze_to_4D",
        "count": 8,
        "args": [
            {
                "k": "t",
                "shape": [32],
                "dtype": "UINT32",
                "layout": "ROW_MAJOR",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "UINT32",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 1, 32],
            },
        ],
    },
    {
        "id": "07_128x2880_bf8_int-dram",
        "op": "ttnn.unsqueeze_to_4D",
        "count": 2,
        "args": [
            {
                "k": "t",
                "shape": [1, 128, 2880],
                "dtype": "BFLOAT8_B",
                "layout": "TILE",
                "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
            },
        ],
        "kwargs": {},
        "outs": [
            {
                "dtype": "BFLOAT8_B",
                "k": "t",
                "layout": "TILE",
                "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
                "shape": [1, 1, 128, 2880],
            },
        ],
    },
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_unsqueeze_to_4D(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
