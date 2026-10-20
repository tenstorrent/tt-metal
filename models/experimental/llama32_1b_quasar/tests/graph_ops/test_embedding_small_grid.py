# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_embedding.py — do NOT regenerate this file
# from a capture.
#
# Every captured ttnn.embedding case is GENERIC + tilized output, so it only exercises
# the fused factory's GENERIC path. It does NOT cover the Quasar self-loop -> scratchpad
# conversions this branch added:
#   * PADDED / BINARY (use_local_cache) -> the WEIGHT_CACHE scratchpad, and
#   * the row-major-output path -> the embeddings_rm factory (whose index_scratch +
#     weight_cache scratchpads no captured case reaches, since the model always tilizes).
#
# These small cases fill that gap. All are tiny (32 tokens x 128-wide table), so they
# fit any grid and finish quickly on the sim. Standard embedding forward is weight[ids]
# for GENERIC/PADDED/BINARY alike (padding_idx only affects the backward pass), so the
# graph_case golden (_ref_embedding) validates all three.
# ---------------------------------------------------------------------------
"""Small-grid ttnn.embedding cases covering the Quasar scratchpad paths (PADDED/BINARY + rm factory)."""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.graph_ops import graph_case as G

_OP = ttnn.embedding
_BINARY = ttnn._ttnn.operations.embedding.EmbeddingsType.BINARY

_DIM = 128  # 4 tiles wide


def _ids_arg():
    return {
        "k": "t",
        "shape": [1, 1, 1, 32],  # 32 tokens (1 tile height, tilizable)
        "dtype": "UINT32",
        "layout": "ROW_MAJOR",
        "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
    }


def _weight_arg(num_emb):
    return {
        "k": "t",
        "shape": [1, 1, num_emb, _DIM],
        "dtype": "BFLOAT16",
        "layout": "ROW_MAJOR",
        "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
    }


def _out(layout):
    return [
        {
            "dtype": "BFLOAT16",
            "k": "t",
            "layout": layout,
            "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
            "shape": [1, 32, _DIM],
        }
    ]


def _case(case_id, num_emb, out_layout, extra_kwargs):
    kwargs = {
        "layout": {"k": "layout", "v": out_layout},
        "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
    }
    kwargs.update(extra_kwargs)
    return {
        "id": case_id,
        "op": "ttnn.embedding",
        "count": 1,
        "args": [_ids_arg(), _weight_arg(num_emb)],
        "kwargs": kwargs,
        "outs": _out(out_layout),
    }


CASES = [
    # fused factory (tilized output) + PADDED -> index_scratch + weight_cache(1 row) scratchpads
    _case("padded_fused_32x128_bf16", 64, "TILE", {"padding_idx": {"k": "lit", "v": 5}}),
    # fused factory + BINARY -> weight_cache(2 rows) scratchpad; 2-row table forces ids in {0,1}
    _case("binary_fused_32x128_bf16", 2, "TILE", {"embeddings_type": {"k": "lit", "v": _BINARY}}),
    # rm factory (row-major output) + GENERIC -> index_scratch scratchpad
    _case("generic_rm_32x128_bf16", 64, "ROW_MAJOR", {}),
    # rm factory + PADDED -> index_scratch + weight_cache(1 row) scratchpads
    _case("padded_rm_32x128_bf16", 64, "ROW_MAJOR", {"padding_idx": {"k": "lit", "v": 5}}),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_embedding_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
