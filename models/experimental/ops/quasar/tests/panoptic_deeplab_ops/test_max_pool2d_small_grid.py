# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated test_max_pool2d.py — do NOT regenerate this file from a capture.
#
# The suite's \`\`-m emulator\`\` subset caps the activation at 48 x 48 x 256 elements, the
# total input footprint at 16 MiB and an L1 shard grid at 8 cores. The captured Panoptic-DeepLab
# call is at 512 x 1024 resolution, so it is outside that subset. The variants below keep the
# call (op arguments, layouts, memory configs, kernel geometry) and shrink the tensors; dtypes
# are bfloat16 directly (the captured bfloat8_b has no Quasar format), so no
# TTNN_GRAPH_OPS_BF8_AS_BF16 is needed.
#
# The captured call is the stem max pool: a 256 x 512 map with 64 channels (8 M elements), 3 x 3
# kernel, stride 2, padding 1, tile output. The variants keep that geometry and use a 16 x 32 map
# (case 00) and a 32 x 32 map (case 01), so the 3 x 3 / stride 2 / padding 1 window is the same
# and the output is 8 x 16 and 16 x 16. The torch golden (max_pool2d) applies unchanged.
# ---------------------------------------------------------------------------
"""Small (2-node) variants of ``ttnn.max_pool2d``: the captured stem max pool on a smaller map."""

import pytest

import ttnn
from models.experimental.ops.quasar.tests.panoptic_deeplab_ops import graph_case as G

_OP = ttnn.max_pool2d

_DRAM = {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None}
_CHANNELS = 64


def _case(case_id, height, width):
    out_h, out_w = (height + 2 - 3) // 2 + 1, (width + 2 - 3) // 2 + 1
    return {
        "id": case_id,
        "op": "ttnn.max_pool2d",
        "count": 1,
        "args": [],
        "kwargs": {
            "input_tensor": {
                "k": "t",
                "shape": [1, 1, height * width, _CHANNELS],
                "dtype": "BFLOAT16",
                "layout": "TILE",
                "mem": dict(_DRAM),
            },
            "channels": {"k": "lit", "v": _CHANNELS},
            "batch_size": {"k": "lit", "v": 1},
            "input_h": {"k": "lit", "v": height},
            "input_w": {"k": "lit", "v": width},
            "kernel_size": {"k": "lit", "v": (3, 3)},
            "stride": {"k": "lit", "v": (2, 2)},
            "padding": {"k": "lit", "v": (1, 1)},
            "dilation": {"k": "lit", "v": (1, 1)},
            "ceil_mode": {"k": "lit", "v": False},
            "deallocate_input": {"k": "lit", "v": False},
            "reallocate_halo_output": {"k": "lit", "v": True},
            "dtype": {"k": "dtype", "v": "BFLOAT16"},
            "output_layout": {"k": "layout", "v": "TILE"},
        },
        "outs": [
            {
                "dtype": "BFLOAT16",
                "k": "t",
                "layout": "TILE",
                "mem": dict(_DRAM),
                "shape": [1, out_h, out_w, _CHANNELS],
            },
        ],
    }


CASES = [
    _case("00_16x32x64_bf16_int-dram", 16, 32),
    _case("01_32x32x64_bf16_int-dram", 32, 32),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_max_pool2d_small_grid(ttnn_mesh_device, reset_seeds, case):
    G.run_case(_OP, case, ttnn_mesh_device)
