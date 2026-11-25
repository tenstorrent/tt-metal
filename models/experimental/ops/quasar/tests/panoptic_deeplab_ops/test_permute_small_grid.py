# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_permute.py — do NOT regenerate this file from a capture.
#
# The suite's \`\`-m emulator\`\` subset caps the activation at 48 x 48 x 256 elements, the
# total input footprint at 16 MiB and an L1 shard grid at 8 cores. The captured Panoptic-DeepLab
# call is at 512 x 1024 resolution, so it is outside that subset. The variants below keep the
# call (op arguments, layouts, memory configs, kernel geometry) and shrink the tensors; dtypes
# are bfloat16 directly (the captured bfloat8_b has no Quasar format), so no
# TTNN_GRAPH_OPS_BF8_AS_BF16 is needed.
#
# The captured calls move the channel dim of an NHWC [1, 128, 256, 32] activation to dim 1
# (permute 0, 3, 1, 2), once as TILE and once as ROW_MAJOR, in interleaved L1. The variants keep
# both layouts and the permutation on a [1, 32, 64, 32] activation. The torch golden applies
# unchanged.
# ---------------------------------------------------------------------------
"""Small (2-node) variants of ``ttnn.permute``: the captured NHWC -> NCHW permutes, reduced."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.panoptic_deeplab_ops import graph_case as G

_OP = ttnn.permute

_L1 = {"layout": "INTERLEAVED", "buffer": "L1", "shard": None}


def _case(case_id, layout):
    return {
        "id": case_id,
        "op": "ttnn.permute",
        "count": 1,
        "args": [
            {"k": "t", "shape": [1, 32, 64, 32], "dtype": "BFLOAT16", "layout": layout, "mem": dict(_L1)},
            {"k": "lit", "v": (0, 3, 1, 2)},
        ],
        "kwargs": {},
        "outs": [
            {"dtype": "BFLOAT16", "k": "t", "layout": layout, "mem": dict(_L1), "shape": [1, 32, 32, 64]},
        ],
    }


CASES = [
    _case("00_32x64x32_bf16_tile_int-l1", "TILE"),
    _case("01_32x64x32_bf16_rm_int-l1", "ROW_MAJOR"),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_permute_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
