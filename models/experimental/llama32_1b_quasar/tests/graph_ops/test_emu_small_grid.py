# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# HAND-AUTHORED (not generated). Scaled-down variants of graph-capture cases
# whose real geometry needs more Tensix cores than the 2-node Quasar emulator
# has, so the captured case is unrunnable there.
#
# The captured nlp_concat_heads_decode case shards its output across
# num_heads=32 cores (compute_output_specs: num_cores_to_corerangeset(num_heads,
# grid)), so on the 1-core emulator it aborts before the op runs:
#   "Target number of cores 32 is greater than total number of available cores 1"
#
# These variants drop num_heads to 1 so the whole op fits on a single core. The
# kernel path (and its Quasar uplift — the borrowed q_out DFB -> LocalTensorAccessor
# conversion) is the same; only the head fan-out shrinks. Run with:
#   pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_emu_small_grid.py -m emulator
# ---------------------------------------------------------------------------
"""Single-core emulator smoke variants of otherwise-too-wide graph_ops cases."""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.graph_ops import graph_case as G

# nlp_concat_heads_decode with a single head: input is one height-sharded shard on
# core (0,0); output is width-sharded across num_heads (=1) cores, i.e. core (0,0).
_NLP_CONCAT_HEADS_DECODE_1HEAD = {
    "id": "1head_32x64_bf16_hs-l1",
    "op": "ttnn.experimental.nlp_concat_heads_decode",
    "count": 1,
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
        "num_heads": {"k": "lit", "v": 1},
    },
    # Output spec unobserved (None) -> graph_case checks count + finiteness and lets
    # the op pin down its own placement, so we don't have to hand-derive the shard grid.
    "outs": [None],
}

CASES = [_NLP_CONCAT_HEADS_DECODE_1HEAD]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_emu_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(ttnn.experimental.nlp_concat_heads_decode, case, ttnn_mesh_device, op_name=case["op"])
