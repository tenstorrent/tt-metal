# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_avg_pool2d.py — do NOT regenerate this file from a capture.
#
# The suite's \`\`-m emulator\`\` subset caps the activation at 48 x 48 x 256 elements, the
# total input footprint at 16 MiB and an L1 shard grid at 8 cores. The captured Panoptic-DeepLab
# call is at 512 x 1024 resolution, so it is outside that subset. The variants below keep the
# call (op arguments, layouts, memory configs, kernel geometry) and shrink the tensors; dtypes
# are bfloat16 directly (the captured bfloat8_b has no Quasar format), so no
# TTNN_GRAPH_OPS_BF8_AS_BF16 is needed.
#
# The captured call is the global average pool before the classifier head: a 32 x 64 map with
# 2048 channels and a (32, 64) kernel (4 M elements). The variants keep a global kernel (the
# kernel covers the whole map, stride 1, no padding) and cut the channels to 256 for the full
# 32 x 64 map, and use a smaller 8 x 16 map. The torch golden (avg_pool2d) applies unchanged.
# ---------------------------------------------------------------------------
"""Small (2-node) variants of ``ttnn.avg_pool2d``: the captured global average pool, reduced."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.panoptic_deeplab_ops import graph_case as G

_OP = ttnn.avg_pool2d

_DRAM = {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None}


def _case(case_id, height, width, channels):
    rows = height * width
    return {
        "id": case_id,
        "op": "ttnn.avg_pool2d",
        "count": 1,
        "args": [],
        "kwargs": {
            "input_tensor": {
                "k": "t",
                "shape": [1, 1, rows, channels],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": dict(_DRAM),
            },
            "batch_size": {"k": "lit", "v": 1},
            "input_h": {"k": "lit", "v": height},
            "input_w": {"k": "lit", "v": width},
            "channels": {"k": "lit", "v": channels},
            "kernel_size": {"k": "lit", "v": (height, width)},
            "stride": {"k": "lit", "v": [1, 1]},
            "padding": {"k": "lit", "v": [0, 0]},
            "ceil_mode": {"k": "lit", "v": False},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "ROW_MAJOR",
                "mem": dict(_DRAM),
                "shape": [1, 1, 1, channels],
            },
        ],
    }


CASES = [
    _case("00_32x64x256_bf16_int-dram", 32, 64, 256),
    _case("01_8x16x256_bf16_int-dram", 8, 16, 256),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_avg_pool2d_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
