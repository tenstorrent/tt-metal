# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_reshard.py — do NOT regenerate this file from a capture.
#
# The suite's \`\`-m emulator\`\` subset caps the activation at 48 x 48 x 256 elements, the
# total input footprint at 16 MiB and an L1 shard grid at 8 cores. The captured Panoptic-DeepLab
# call is at 512 x 1024 resolution on a 20-core device, so it is outside that subset. The variants
# below keep the call and shrink the tensors; dtypes are bfloat16 directly (the captured
# bfloat8_b has no Quasar format), so no TTNN_GRAPH_OPS_BF8_AS_BF16 is needed.
#
# The captured call reshards a [1, 1, 2048, 1024] activation from BLOCK_SHARDED on a 5 x 4 grid to
# HEIGHT_SHARDED on 19 cores. The variants keep that conversion, and its reverse, on a 2 x 1 grid
# ([[0, 0, 1, 0]]) with a [1, 1, 128, 128] activation: BLOCK_SHARDED 128 x 64 shards to
# HEIGHT_SHARDED 64 x 128 shards, and back. The golden is the identity.
# ---------------------------------------------------------------------------
"""Small (2-node) variants of ``ttnn.reshard``: block <-> height resharding on a 2 x 1 grid."""

import copy

import pytest

import ttnn
from models.experimental.ops.quasar.tests.panoptic_deeplab_ops import graph_case as G

_OP = ttnn.reshard

_GRID = [[0, 0, 1, 0]]
_SHAPE = [1, 1, 128, 128]


def _mem(layout, shard_shape):
    return {
        "layout": layout,
        "buffer": "L1",
        "shard": {"grid": copy.deepcopy(_GRID), "shape": list(shard_shape), "orientation": "ROW_MAJOR"},
    }


_BLOCK = _mem("BLOCK_SHARDED", [128, 64])
_HEIGHT = _mem("HEIGHT_SHARDED", [64, 128])


def _case(case_id, src, dst):
    return {
        "id": case_id,
        "op": "ttnn.reshard",
        "count": 1,
        "args": [
            {"k": "t", "shape": list(_SHAPE), "dtype": "BFLOAT16", "layout": "TILE", "mem": copy.deepcopy(src)},
            {**copy.deepcopy(dst), "k": "mem"},
        ],
        "kwargs": {},
        "outs": [
            {"dtype": "BFLOAT16", "k": "t", "layout": "TILE", "mem": copy.deepcopy(dst), "shape": list(_SHAPE)},
        ],
    }


CASES = [
    _case("00_128x128_bf16_bs_to_hs-l1", _BLOCK, _HEIGHT),
    _case("01_128x128_bf16_hs_to_bs-l1", _HEIGHT, _BLOCK),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_reshard_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
