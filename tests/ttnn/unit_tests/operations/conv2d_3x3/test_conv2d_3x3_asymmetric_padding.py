# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3 per-side padding (top, bottom, left, right), each side 0 or dilation, on DRAM-interleaved input.

Rows move only with the top side and columns only with the left; bottom / right set the output extent and how far
the last row's / column's taps reach past the image. The cases put the zero padding on every side alone and in pairs,
on the paths the interleaved walk takes: the TILE ring / compacted slab and the ROW_MAJOR slab, tap-major (stride 1),
the strided window (stride 2, W_out a multiple of 32) and the row gather (stride 2 otherwise, dilation 2, odd W).
Sharded input refuses per-side padding (EXCLUSIONS).
"""

import importlib

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

_op = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3")

PCC = {ttnn.bfloat16: 0.999, ttnn.bfloat8_b: 0.99}

# (id, (H, W, C_in), C_out, (top, bottom, left, right), stride, dilation)
CASES = [
    ("rows_same_cols_valid", (16, 34, 32), 32, (1, 1, 0, 0), 1, 1),
    ("top_left_odd_w", (32, 33, 32), 64, (1, 0, 1, 0), 1, 1),
    ("bottom_left", (33, 32, 32), 32, (0, 1, 1, 0), 1, 1),
    ("halo_rows_multi_block", (66, 128, 64), 128, (0, 0, 1, 1), 1, 1),
    ("s2_bottom_right_window", (32, 64, 64), 64, (0, 1, 0, 1), 2, 1),
    ("s2_top_right", (9, 64, 32), 32, (1, 0, 0, 1), 2, 1),
    ("d2_cols_same", (8, 48, 32), 32, (0, 0, 2, 2), 1, 2),
    ("d2_top_right", (16, 34, 32), 32, (2, 0, 0, 2), 1, 2),
]

DTYPE_LAYOUTS = [
    (ttnn.bfloat16, ttnn.TILE_LAYOUT),
    (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
]


def _reference(x_hwc, weight, bias, padding, stride, dilation):
    top, bottom, left, right = padding
    x = torch.nn.functional.pad(x_hwc.float().permute(2, 0, 1).unsqueeze(0), (left, right, top, bottom))
    out = torch.nn.functional.conv2d(x, weight.float(), bias.float(), stride=stride, dilation=dilation)
    return out[0].permute(1, 2, 0).reshape(1, 1, -1, weight.shape[0])


@pytest.mark.parametrize("dtype,layout", DTYPE_LAYOUTS, ids=["bf16_tile", "bf8b_tile", "bf16_rm"])
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_conv2d_3x3_asymmetric_padding(device, case, dtype, layout):
    _, (H, W, c_in), c_out, padding, stride, dilation = case
    torch.manual_seed(0)
    x = torch.randn(H, W, c_in, dtype=torch.bfloat16)
    weight = (torch.randn(c_out, c_in, 3, 3) / (9 * c_in) ** 0.5).to(torch.bfloat16)
    bias = (0.1 * torch.randn(c_out)).to(torch.bfloat16)

    prep = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=ttnn.bfloat16,
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
    )
    tt_w = _op.prepare_conv2d_weights(weight, **prep)
    tt_b = _op.prepare_conv2d_bias(bias, **prep)
    tt_x = ttnn.from_torch(
        x.reshape(1, 1, H * W, c_in), dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = _op.conv2d_3x3(
        tt_x, tt_w, input_height=H, input_width=W, bias=tt_b, padding=padding, stride=stride, dilation=dilation
    )

    expected = _reference(x, weight, bias, padding, stride, dilation)
    h_out, w_out = _op.output_hw(H, W, padding=padding, stride=stride, dilation=dilation)
    assert expected.shape[2] == h_out * w_out
    got = ttnn.to_torch(out)
    assert list(got.shape) == [1, 1, h_out * w_out, c_out]
    assert_with_pcc(expected, got.float(), PCC[dtype])


def test_conv2d_3x3_asymmetric_padding_sharded_refused(device, expect_error):
    """Per-side padding on a sharded input is an EXCLUSIONS cell: prepare_* refuses it before anything is uploaded."""
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))])
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [128, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    with expect_error(_op.ExcludedCell, "asymmetric"):
        _op.prepare_conv2d_weights(
            torch.zeros(32, 32, 3, 3),
            padding=(0, 0, 1, 1),
            input_dtype=ttnn.bfloat16,
            input_layout=ttnn.TILE_LAYOUT,
            weight_dtype=ttnn.bfloat16,
            input_memory_config=memory_config,
            device=device,
        )
