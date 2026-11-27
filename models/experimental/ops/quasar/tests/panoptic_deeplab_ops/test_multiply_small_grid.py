# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_multiply.py — do NOT regenerate this file from a capture.
#
# The suite's \`\`-m emulator\`\` subset caps the activation at 48 x 48 x 256 elements, the
# total input footprint at 16 MiB and an L1 shard grid at 8 cores. The captured Panoptic-DeepLab
# call is at 512 x 1024 resolution, so it is outside that subset. The variants below keep the
# call (op arguments, layouts, memory configs, kernel geometry) and shrink the tensors; dtypes
# are bfloat16 directly (the captured bfloat8_b has no Quasar format), so no
# TTNN_GRAPH_OPS_BF8_AS_BF16 is needed.
#
# The captured call multiplies a [1, 32, 512, 1024] activation (16 M elements) by the scalar 4.
# The variants keep the scalar multiply on a [1, 4, 64, 256] and a [1, 1, 32, 128] activation in
# interleaved DRAM. The torch golden (mul) applies unchanged.
# ---------------------------------------------------------------------------
"""Small (2-node) variants of ``ttnn.multiply``: the captured scalar multiply, reduced."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.panoptic_deeplab_ops import graph_case as G

_OP = ttnn.multiply

_DRAM = {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None}


def _case(case_id, shape):
    return {
        "id": case_id,
        "op": "ttnn.multiply",
        "count": 1,
        "args": [
            {"k": "t", "shape": shape, "dtype": "BFLOAT16", "layout": "TILE", "mem": dict(_DRAM)},
            {"k": "lit", "v": 4},
        ],
        "kwargs": {},
        "outs": [{"dtype": "BFLOAT16", "k": "t", "layout": "TILE", "mem": dict(_DRAM), "shape": shape}],
    }


CASES = [
    _case("00_4x64x256_bf16_int-dram", [1, 4, 64, 256]),
    _case("01_1x32x128_bf16_int-dram", [1, 1, 32, 128]),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_multiply_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
