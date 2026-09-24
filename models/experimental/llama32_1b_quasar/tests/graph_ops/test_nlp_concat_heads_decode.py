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
Per-op test: ``ttnn.experimental.nlp_concat_heads_decode`` — every distinct call the model made, as captured.

Captured 307 call(s) to this op; 1 distinct signature(s) covering 307 of them.

Fidelity notes for this op:
  * the output has more elements than all inputs combined (a batch-padded decode tensor), so the op cannot write all of it; finiteness is asserted over the portion the inputs can account for

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

_OP = ttnn.experimental.nlp_concat_heads_decode

CASES = [
    {
        "id": "00_32x64_bf16_hs-l1",
        "op": "ttnn.experimental.nlp_concat_heads_decode",
        "count": 307,
        "args": [
            {
                "k": "t",
                "shape": [1, 1, 32, 64],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": {
                    "layout": "HEIGHT_SHARDED",
                    "buffer": "L1",
                    "shard": {"grid": [[0, 0, 0, 0]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
                },
            },
        ],
        "kwargs": {
            "num_heads": {"k": "lit", "v": 32},
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
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_nlp_concat_heads_decode(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
