# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 8: channel alignment, dilation, grouped convs.

Golden-style cases through the golden runner (prepare_* + PCC / rel-RMS gate), one per kernel path:
unaligned C_in / C_out / both (R0 RM incl. the staged DRAM window, TILE DRAM, HEIGHT, BLOCK),
dilation (R0, HEIGHT, BLOCK, depthwise), grouped (native per-group R0, block-diagonal R0 / HEIGHT).
"""

import pytest
import ttnn

from eval.golden_tests.conv2d_nhwc.helpers import run_conv2d_nhwc

ML = ttnn.TensorMemoryLayout
RM, TL = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
BF16, FP32, BF8 = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b

CASES = [
    # (inputs, dtype, weight_dtype, layout, memory_layout, extras)
    (((1, 8, 8, 3), 32, 3, 1, 1, 1, 1), BF16, BF16, RM, ML.INTERLEAVED, None),  # C_in 3, staged window
    (((1, 14, 14, 1), 32, 5, 2, 1, 1, 1), FP32, FP32, RM, ML.INTERLEAVED, None),  # C_in 1 fp32
    (((1, 14, 14, 32), 47, 3, 0, 1, 1, 1), BF16, BF16, RM, ML.INTERLEAVED, None),  # C_out 47
    (((1, 16, 16, 17), 33, 3, 0, 1, 1, 1), FP32, FP32, RM, ML.INTERLEAVED, None),  # both unaligned
    (((1, 224, 224, 3), 64, 7, 3, 2, 1, 1), BF16, BF16, RM, ML.INTERLEAVED, None),  # ResNet conv1
    (((1, 28, 28, 16), 16, 3, 1, 1, 16, 1), BF16, BF16, RM, ML.INTERLEAVED, None),  # half-tile depthwise
    (((1, 28, 28, 48), 48, 3, 1, 1, 48, 1), FP32, FP32, RM, ML.INTERLEAVED, None),  # 1.5-tile depthwise
    (((1, 32, 32, 64), 64, 3, 4, 1, 1, 4), BF16, BF16, RM, ML.INTERLEAVED, None),  # dilated
    (((1, 28, 28, 17), 32, 3, 2, 1, 1, 2), FP32, FP32, RM, ML.INTERLEAVED, None),  # dilated + C_in 17
    (((1, 28, 28, 96), 96, 3, 2, 1, 96, 2), BF16, BF16, RM, ML.INTERLEAVED, None),  # dilated depthwise
    (((1, 13, 13, 384), 256, 3, 1, 1, 2, 1), BF16, BF16, RM, ML.INTERLEAVED, None),  # native groups
    (((1, 28, 28, 128), 128, 3, 1, 1, 32, 1), BF16, BF16, RM, ML.INTERLEAVED, None),  # ResNeXt g=32 (block-diag)
    (((1, 56, 56, 64), 128, 3, 2, 1, 4, 2), FP32, FP32, RM, ML.INTERLEAVED, None),  # atrous + grouped
    (((1, 64, 64, 4), 64, 3, 1, 1, 1, 1, ML.INTERLEAVED, TL), BF16, BF16, TL, ML.INTERLEAVED, None),  # TILE C_in 4
    (((1, 224, 224, 3), 64, 7, 3, 2, 1, 1, ML.HEIGHT_SHARDED), BF16, BF16, RM, ML.HEIGHT_SHARDED, None),
    (((1, 28, 28, 48), 80, 3, 1, 1, 1, 1, ML.HEIGHT_SHARDED), BF16, BF16, RM, ML.HEIGHT_SHARDED, None),
    (((1, 64, 64, 64), 64, 3, 4, 1, 1, 4, ML.HEIGHT_SHARDED), BF16, BF16, RM, ML.HEIGHT_SHARDED, None),
    (((1, 28, 28, 64), 64, 3, 1, 1, 4, 1, ML.HEIGHT_SHARDED), BF16, BF16, RM, ML.HEIGHT_SHARDED, None),
    (((1, 32, 32, 128), 128, 3, 2, 1, 1, 2, ML.BLOCK_SHARDED), BF16, BF16, RM, ML.BLOCK_SHARDED, None),
    (
        ((1, 128, 128, 4), 320, 3, 1, 1, 1, 1, ML.HEIGHT_SHARDED),
        BF16,
        BF8,
        RM,
        ML.HEIGHT_SHARDED,
        {"shard_shape": [160, 8], "num_cores": 103},
    ),  # SDXL conv_in
    (
        ((1, 128, 128, 320), 4, 3, 1, 1, 1, 1, ML.HEIGHT_SHARDED),
        BF16,
        BF8,
        RM,
        ML.HEIGHT_SHARDED,
        {"shard_shape": [160, 320], "num_cores": 103},
    ),  # SDXL conv_out
    (
        ((1, 128, 128, 320), 4, 3, 1, 1, 1, 1, ML.BLOCK_SHARDED),
        BF16,
        BF16,
        RM,
        ML.BLOCK_SHARDED,
        {"shard_shape": [2048, 40], "core_grid": (8, 8)},
    ),  # conv_out BLOCK
]


def _id(c):
    (N, H, W, Ci), Co, k, p, s, g, d = c[0][:7]
    return f"{H}x{W}x{Ci}_Co{Co}_k{k}p{p}s{s}g{g}d{d}_{c[4].name[:5]}_{c[3].name[:4]}_{c[1].name}"


@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
def test_conv2d_nhwc_r8(device, case):
    inputs, dtype, wdtype, layout, ml, extras = case
    run_conv2d_nhwc(
        inputs,
        dtype=dtype,
        weight_dtype=wdtype,
        layout=layout,
        bias_mode="with_bias",
        memory_layout=ml,
        device=device,
        extras=extras,
    )
