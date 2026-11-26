# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_slice.py — do NOT regenerate this file from a capture.
#
# The suite's \`\`-m emulator\`\` subset caps the activation at 48 x 48 x 256 elements, the
# total input footprint at 16 MiB and an L1 shard grid at 8 cores. The captured Panoptic-DeepLab
# call is at 512 x 1024 resolution, so it is outside that subset. The variants below keep the
# call (op arguments, layouts, memory configs, kernel geometry) and shrink the tensors; dtypes
# are bfloat16 directly (the captured bfloat8_b has no Quasar format), so no
# TTNN_GRAPH_OPS_BF8_AS_BF16 is needed.
#
# The captured calls cut 512-wide column windows out of a [1, 1, 2048, 2048] and a
# [1, 32, 64, 2048] interleaved DRAM tensor (the ASPP / decoder channel splits). The variants keep
# the interleaved DRAM TILE slices along the last dim and use a [1, 1, 256, 512] and a
# [1, 4, 64, 512] tensor with 128-wide windows at offsets 0, 128 and 384. The torch golden
# (slice) applies unchanged.
# ---------------------------------------------------------------------------
"""Small (2-node) variants of ``ttnn.slice``: the captured channel-window slices, reduced."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.panoptic_deeplab_ops import graph_case as G

_OP = ttnn.slice

_DRAM = {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None}


def _case(case_id, shape, start, width):
    out_shape = list(shape[:-1]) + [width]
    end = list(shape[:-1]) + [start + width]
    return {
        "id": case_id,
        "op": "ttnn.slice",
        "count": 1,
        "args": [
            {"k": "t", "shape": shape, "dtype": "BFLOAT16", "layout": "TILE", "mem": dict(_DRAM)},
            {"k": "lit", "v": [0, 0, 0, start]},
            {"k": "lit", "v": end},
        ],
        "kwargs": {},
        "outs": [{"dtype": "BFLOAT16", "k": "t", "layout": "TILE", "mem": dict(_DRAM), "shape": out_shape}],
    }


CASES = [
    _case("00_256x512_bf16_cols0_int-dram", [1, 1, 256, 512], 0, 128),
    _case("01_256x512_bf16_cols128_int-dram", [1, 1, 256, 512], 128, 128),
    _case("02_256x512_bf16_cols384_int-dram", [1, 1, 256, 512], 384, 128),
    _case("03_4x64x512_bf16_cols128_int-dram", [1, 4, 64, 512], 128, 128),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_slice_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
