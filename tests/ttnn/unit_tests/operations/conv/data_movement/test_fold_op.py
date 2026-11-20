# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import re

import pytest
import torch

import ttnn
import math

from tt_py_test_utils_common.utility_functions import (
    _nearest_y,
    is_wormhole_b0,
    torch2tt_tensor,
    tt2torch_tensor,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.data_movement.docstring_support import (
    UnreadableDocstring,
    check_readable,
    docstring_dtype_layout_params,
    docstring_interleaved_buffer_type_params,
    interleaved_memory_config,
)
from tests.ttnn.unit_tests.operations.data_movement.data_movement_test_utils import assert_exact, full_range_input


def pad_and_fold_conv_activation_for_unity_stride(activation_pyt_nchw_tensor, pad_h, pad_w, stride_h, stride_w):
    assert stride_h == stride_w
    assert activation_pyt_nchw_tensor.shape[2] == activation_pyt_nchw_tensor.shape[3]
    # Fold activation for unity stride
    # Pad channel size to 4. This is to make sure L1 read addresses are 16 bit aligned
    C = _nearest_y(activation_pyt_nchw_tensor.shape[1], 4)
    # Also, pre-pad the conv left right and top bottom padding
    activation_pyt_padded = torch.nn.functional.pad(
        activation_pyt_nchw_tensor, (pad_w, pad_w, pad_h, pad_h, 0, C - activation_pyt_nchw_tensor.shape[1])
    )
    # Fold the activation face by stride depth wise i.e. C,H,W -> C*stride_h*stride_w, H/stride_h, W/stride_w
    assert activation_pyt_padded.shape[2] % stride_h == 0
    activation_pyt_padded_folded = torch.zeros(
        [
            activation_pyt_padded.shape[0],
            C * stride_h * stride_w,
            (int)(activation_pyt_padded.shape[2] / stride_h),
            (int)(activation_pyt_padded.shape[3] / stride_w),
        ]
    )
    for h in range(0, activation_pyt_padded.shape[2], stride_h):
        for w in range(0, activation_pyt_padded.shape[3], stride_w):
            folded_h = (int)(h / stride_h)
            folded_w = (int)(w / stride_w)
            for i in range(stride_h * stride_w):
                start_c = i * C
                activation_pyt_padded_folded[:, start_c : start_c + C, folded_h, folded_w] = activation_pyt_padded[
                    :, :, h + (int)(i / stride_w), w + (int)(i % stride_w)
                ]

    return activation_pyt_padded_folded


def fold_torch(input_tensor, stride_h, stride_w, padding=None):
    N, H, W, C = input_tensor.shape

    # Handle padding specification
    if padding is not None:
        if len(padding) == 2:
            # Symmetric padding: [pad_h, pad_w]
            pad_top = pad_bottom = padding[0]
            pad_left = pad_right = padding[1]
        elif len(padding) == 4:
            # Asymmetric padding: [top, bottom, left, right]
            pad_top, pad_bottom, pad_left, pad_right = padding
        else:
            raise ValueError(f"Padding must be length 2 or 4, got {len(padding)}")

        # Apply padding if needed
        if pad_top != 0 or pad_bottom != 0 or pad_left != 0 or pad_right != 0:
            import torch.nn.functional as F

            input_tensor = F.pad(input_tensor, (0, 0, pad_left, pad_right, pad_top, pad_bottom), value=0.0)
            N, H, W, C = input_tensor.shape

    reshaped = input_tensor.reshape(N, H // stride_h, stride_h, W // stride_w, stride_w, C)
    transposed = reshaped.permute(0, 1, 3, 2, 4, 5)
    return transposed.reshape(N, H // stride_h, W // stride_w, C * stride_h * stride_w)


@pytest.mark.parametrize("nhw", [(3, 64, 64), (1, 224, 224), (1, 384, 512), (1, 512, 672)])
@pytest.mark.parametrize("channels", [3, 32, 320])
@pytest.mark.parametrize("stride", [(16, 16), (32, 32)])
@pytest.mark.parametrize("padding", [(0, 0), (8, 8), (20, 12, 15, 17)])
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32])
def test_fold_with_permute_for_dram_tensor(device, nhw, channels, stride, padding, input_layout, input_dtype):
    batch_size, height, width = nhw
    stride_h, stride_w = stride

    if input_dtype == ttnn.bfloat8_b and input_layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("bfloat8_b with row major layout is not supported")
    # Handle both symmetric and asymmetric padding
    if len(padding) == 2:
        # Symmetric padding: [pad_h, pad_w]
        padded_h = height + 2 * padding[0]
        padded_w = width + 2 * padding[1]
        has_padding = padding[0] > 0 or padding[1] > 0
    elif len(padding) == 4:
        # Asymmetric padding: [top, bottom, left, right]
        padded_h = height + padding[0] + padding[1]
        padded_w = width + padding[2] + padding[3]
        has_padding = any(p > 0 for p in padding)
    else:
        pytest.skip(f"Invalid padding format: {padding}")

    # Skip invalid combinations where padding makes dimensions not divisible by stride
    if has_padding and input_layout == ttnn.TILE_LAYOUT:
        pytest.skip("ttnn::pad with tile layout does not support front padding yet")
    if padded_h % stride_h != 0 or padded_w % stride_w != 0:
        pytest.skip(
            f"Skipping invalid padding combination: padded_h={padded_h}, padded_w={padded_w}, stride_h={stride_h}, stride_w={stride_w}"
        )

    # fp32 + channels=320 + stride=(32,32): output stick alone is 1.25 MB > per-core L1 — device capacity, not a bug.
    if input_dtype == ttnn.float32 and channels == 320 and stride == (32, 32):
        pytest.skip("FP32 fold: channels=320 + 32x32 stride overflows per-core L1 (capacity limit)")

    torch_input_dtype = torch.float32 if input_dtype == ttnn.float32 else torch.bfloat16
    torch_input_tensor = torch.rand((batch_size, channels, height, width), dtype=torch_input_dtype)
    torch_input_tensor_nhwc = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    torch_output_tensor = fold_torch(torch_input_tensor_nhwc, stride_h, stride_w, padding=padding)
    input_memory_config = ttnn.DRAM_MEMORY_CONFIG
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor_nhwc, input_dtype, layout=input_layout, device=device, memory_config=input_memory_config
    )
    tt_output_tensor = ttnn.fold(
        tt_input_tensor,
        stride_h=stride_h,
        stride_w=stride_w,
        padding=list(padding),
    )
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)
    if input_dtype == ttnn.bfloat8_b:
        threshold = 5e-4 * math.log(batch_size * channels * height * width, 2) + 1e-3
        diff = torch.abs(torch_output_tensor - tt_output_tensor) / torch_output_tensor.abs().mean()
        assert torch.all(diff < threshold), f"Max diff: {diff.max()}, Threshold: {threshold} "
    else:
        # bfloat16 and float32 are both bit-exact on this path (the reference and the device
        # output share the input dtype, and fold only moves elements), so assert exact equality.
        #
        # Scope: this parametrization runs on DRAM (input_memory_config above). The guarantee is
        # not general - an L1 height-sharded TILE input with padding goes through halo's untilize
        # instead, which unpacks a Float32 input via SrcA as Tf32 and loses mantissa bits. That is
        # intended: halo is a staging op whose consumers narrow to Tf32 anyway. See #60778,
        # #58402 item 14 and #60731.
        assert torch.equal(
            torch_output_tensor, tt_output_tensor
        ), f"Expected: {torch_output_tensor}, Actual: {tt_output_tensor}"


def pad_and_fold_with_permute_and_reshape_on_device(
    device, activation_pyt_nchw_tensor, pad_h, pad_w, stride_h, stride_w
):
    # pad on host
    n, c, h, w = activation_pyt_nchw_tensor.shape
    padded_h = h + pad_h * 2
    padded_w = w + pad_w * 2
    w_pad32 = padded_w + (32 - padded_w % 32) % 32
    target_h = padded_h // stride_h
    target_w = padded_w // stride_w
    C = _nearest_y(c, 4)
    pad_w_right = w_pad32 - (w + pad_w)
    activation_pyt_padded = torch.nn.functional.pad(activation_pyt_nchw_tensor, (pad_w, pad_w_right, pad_h, pad_h))
    # pad on device to 256, 256
    n, c, h, w = activation_pyt_padded.shape
    padding = ((0, C - c), (0, w_pad32 - h), (0, 0))
    activation_pyt_padded = ttnn.from_torch(
        activation_pyt_padded, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    activation_pyt_padded = ttnn.pad(
        activation_pyt_padded, padding=padding, value=0, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    # transpose
    activation_pyt_padded = ttnn.transpose(activation_pyt_padded, 2, 3, memory_config=ttnn.L1_MEMORY_CONFIG)
    # transpose
    activation_pyt_padded = ttnn.transpose(activation_pyt_padded, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    # reshape
    n, w, c, h = activation_pyt_padded.shape
    activation_pyt_padded = ttnn.reshape_on_device(
        activation_pyt_padded, n, w // stride_w, c * stride_w, h, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    # transpose
    activation_pyt_padded = ttnn.transpose(activation_pyt_padded, 2, 3, memory_config=ttnn.L1_MEMORY_CONFIG)
    # reshape
    n, w, h, c = activation_pyt_padded.shape
    activation_pyt_padded = ttnn.reshape_on_device(
        activation_pyt_padded, n, w, h // stride_h, c * stride_h, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    # transpose
    activation_pyt_padded = ttnn.transpose(activation_pyt_padded, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
    # slice
    n, h, w, c = activation_pyt_padded.shape
    activation_pyt_padded = ttnn.slice(
        activation_pyt_padded,
        (0, 0, 0, 0),
        (n, target_h, target_w, c),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    return activation_pyt_padded


def pad_and_fold_with_permute_and_reshape_on_device_sharded(device, tt_input_tensor, pad_h, pad_w, stride_h, stride_w):
    n, c, h, w = tt_input_tensor.shape
    C = _nearest_y(c, 4)
    padded_h = h + pad_h * 2
    padded_w = w + pad_w * 2
    w_pad32 = padded_w + (32 - padded_w % 32) % 32
    h_pad32 = padded_h + (32 - padded_h % 32) % 32
    pad_w_right = w_pad32 - (w + pad_w)
    pad_h_right = h_pad32 - (h + pad_h)

    target_h = (h + pad_h * 2) // stride_h
    target_w = (w + pad_w * 2) // stride_w
    ##################################### pad on device to 256, 224 #####################################
    activation_pyt_padded_shape = [n, C, h_pad32, w]
    pad_sharded_memory_config = ttnn.create_sharded_memory_config(
        activation_pyt_padded_shape,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    print("pad " + str(tt_input_tensor.shape))
    tt_output_tensor = ttnn.pad(
        tt_input_tensor,
        padding=((0, C - c), (pad_h, pad_h_right), (0, 0)),
        value=0,
        memory_config=pad_sharded_memory_config,
    )
    ##################################### transpose ######################################################
    tphw_sharded_memory_config = ttnn.create_sharded_memory_config(
        tt_output_tensor.shape,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    print("transpose hw " + str(tt_output_tensor.shape))
    tt_output_tensor = ttnn.transpose(tt_output_tensor, 2, 3, memory_config=tphw_sharded_memory_config)
    ##################################### pad on device to 256, 256 #####################################
    activation_pyt_padded_shape = [n, C, h_pad32, w_pad32]
    pad_sharded_memory_config = ttnn.create_sharded_memory_config(
        activation_pyt_padded_shape,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    print("pad " + str(tt_output_tensor.shape))
    tt_output_tensor = ttnn.pad(
        tt_output_tensor,
        padding=((0, 0), (pad_w, pad_w_right), (0, 0)),
        value=0,
        memory_config=pad_sharded_memory_config,
    )
    ##################################### transpose ######################################################
    tphc_sharded_memory_config = ttnn.create_sharded_memory_config(
        tt_output_tensor.shape,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    print("transpose hc " + str(tt_output_tensor.shape))
    tt_output_tensor = ttnn.transpose(tt_output_tensor, 1, 2, memory_config=tphc_sharded_memory_config)
    ##################################### reshape ######################################################
    n, w, c, h = tt_output_tensor.shape
    tt_output_tensor = tt_output_tensor.reshape(n, w // stride_w, c * stride_w, h)
    ##################################### transpose ####################################################
    tphw_sharded_memory_config = ttnn.create_sharded_memory_config(
        tt_output_tensor.shape,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    print("transpose hw " + str(tt_output_tensor.shape))
    tt_output_tensor = ttnn.transpose(tt_output_tensor, 2, 3, memory_config=tphw_sharded_memory_config)
    ##################################### reshape #######################################################
    n, w, h, c = tt_output_tensor.shape
    tt_output_tensor = tt_output_tensor.reshape(n, w, h // stride_h, c * stride_h)
    ##################################### transpose #####################################################
    tphc_sharded_memory_config = ttnn.create_sharded_memory_config(
        tt_output_tensor.shape,
        core_grid=ttnn.CoreGrid(y=8, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    print("transpose hc " + str(tt_output_tensor.shape))
    tt_output_tensor = ttnn.transpose(tt_output_tensor, 1, 2, memory_config=tphc_sharded_memory_config)
    ##################################### slice #########################################################
    n, h, w, c = tt_output_tensor.shape
    num_cores_x = 8
    num_cores_y = 8
    shard_h = (n * target_h * target_w + (num_cores_x * num_cores_y) - 1) // (num_cores_x * num_cores_y)
    grid_size = ttnn.CoreGrid(y=num_cores_y, x=num_cores_x)
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, c), ttnn.ShardOrientation.ROW_MAJOR)
    slice_sharded_memory_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    tt_output_tensor = ttnn.slice(
        tt_output_tensor,
        (0, 0, 0, 0),
        (n, target_h, target_w, c),
        memory_config=slice_sharded_memory_config,
    )
    print("output " + str(tt_output_tensor.shape))
    return tt_output_tensor


@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [224])
@pytest.mark.parametrize("w", [224])
@pytest.mark.parametrize("pad_h", [3])
@pytest.mark.parametrize("pad_w", [3])
@pytest.mark.parametrize("stride_h", [2])
@pytest.mark.parametrize("stride_w", [2])
def test_fold_with_permute_reshape_on_device_sharded(device, n, c, h, w, pad_h, pad_w, stride_h, stride_w):
    if device.core_grid.y < 8:
        pytest.skip("n300 does not have 8x8 grid")
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = pad_and_fold_conv_activation_for_unity_stride(
        torch_input_tensor, pad_h, pad_w, stride_h, stride_w
    )
    torch_output_tensor = torch.permute(torch_output_tensor, (0, 2, 3, 1))
    # on device
    in_sharded_memory_config = ttnn.create_sharded_memory_config(
        torch_input_tensor.shape,
        core_grid=ttnn.CoreGrid(y=8, x=6),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_sharded_memory_config
    )
    grid_size = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    tt_output_tensor = ttnn.fold(
        tt_input_tensor,
        stride_h,
        stride_w,
        use_transpose_as_fold=True,
        padding=[pad_h, pad_h, pad_w, pad_w, 0, _nearest_y(c, 4) - c],
        grid_size=grid_size,
    )
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)
    assert_with_pcc(torch_output_tensor, tt_output_tensor, 1)


@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [224])
@pytest.mark.parametrize("w", [224])
@pytest.mark.parametrize("pad_h", [3])
@pytest.mark.parametrize("pad_w", [3])
@pytest.mark.parametrize("stride_h", [2])
@pytest.mark.parametrize("stride_w", [2])
def test_fold_with_permute_reshape_on_device(device, n, c, h, w, pad_h, pad_w, stride_h, stride_w):
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = pad_and_fold_conv_activation_for_unity_stride(
        torch_input_tensor, pad_h, pad_w, stride_h, stride_w
    )
    torch_output_tensor = torch.permute(torch_output_tensor, (0, 2, 3, 1))
    # pad on host
    n, c, h, w = torch_input_tensor.shape
    C = _nearest_y(c, 4)
    padded_h = h + pad_h * 2
    padded_w = w + pad_w * 2
    w_pad32 = padded_w + (32 - padded_w % 32) % 32
    pad_w_right = w_pad32 - (w + pad_w)
    torch_input_tensor_padded = torch.nn.functional.pad(torch_input_tensor, (pad_w, pad_w_right, pad_h, pad_h))
    # on device
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor_padded, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_output_tensor = ttnn.fold(
        tt_input_tensor,
        stride_h,
        stride_w,
        use_transpose_as_fold=True,
        output_shape=(n, padded_h // stride_h, padded_w // stride_w, C * (stride_h * stride_w)),
        padding=[0, 0, 0, 0, 0, C - c],
    )
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)
    assert_with_pcc(torch_output_tensor, tt_output_tensor, 1)


@pytest.mark.parametrize(
    "act_shape,stride_h,stride_w",
    [
        ((1, 2, 2, 16), 2, 2),
        ((10, 2, 2, 32), 2, 2),
        ((10, 4, 4, 32), 2, 2),
        ((10, 6, 8, 32), 3, 2),
        ((10, 6, 8, 32), 3, 1),
        ((10, 6, 8, 32), 1, 2),
        ((10, 6, 8, 32), 1, 1),
        ((1, 4, 2, 8), 2, 1),
    ],
)
def test_fold(act_shape, stride_h, stride_w, device):
    torch.manual_seed(0)

    torch_input = torch.randn(act_shape, dtype=torch.bfloat16)

    expected = fold_torch(torch_input, stride_h, stride_w)

    tt_input = torch2tt_tensor(
        torch_input,
        device,
        ttnn.ROW_MAJOR_LAYOUT,
        tt_memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_out = ttnn.fold(tt_input, stride_h, stride_w)
    actual = tt2torch_tensor(tt_out)

    torch.testing.assert_close(actual, expected)


def _fold_reference(x, stride_h, stride_w, padding=(0, 0), collapse_output=False):
    """torch fold of an [N, H, W, C] tensor after the zero padding ttnn.fold takes: [pad_h, pad_w],
    [top, bottom, left, right] or [top, bottom, left, right, c_front, c_back]. Computed in float64, which holds every
    value of the dtypes under test exactly."""
    if len(padding) == 2:
        padding = [padding[0], padding[0], padding[1], padding[1], 0, 0]
    elif len(padding) == 4:
        padding = list(padding) + [0, 0]
    top, bottom, left, right, c_front, c_back = padding
    x = torch.nn.functional.pad(x.to(torch.float64), (c_front, c_back, left, right, top, bottom))
    n, h, w, c = x.shape
    out = x.reshape(n, h // stride_h, stride_h, w // stride_w, stride_w, c).permute(0, 1, 3, 2, 4, 5)
    out = out.reshape(n, h // stride_h, w // stride_w, c * stride_h * stride_w)
    return out.reshape(1, 1, -1, out.shape[-1]) if collapse_output else out


def _fold_output_dtype(dtype):
    return ttnn.bfloat16 if dtype == ttnn.bfloat8_b else dtype


# (shape, stride_h, stride_w, padding). With C=16 a TILE input takes the tile-native path, with C=3 the untilize one.
_DOCSTRING_CASES = [((2, 8, 12, 16), 2, 2, [0, 0]), ((1, 6, 9, 3), 3, 3, [0, 0])]
# A ROW_MAJOR input also takes each padding form.
_DOCSTRING_PADDED_CASES = [
    ((1, 4, 2, 8), 2, 3, [1, 2]),
    ((2, 4, 5, 8), 3, 2, [0, 2, 1, 0]),
    ((1, 5, 6, 4), 2, 3, [1, 0, 2, 1, 1, 3]),
]


@pytest.mark.parametrize("buffer_type", docstring_interleaved_buffer_type_params(ttnn.fold))
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.fold))
def test_fold_docstring_supported_inputs(dtype, layout, buffer_type, device):
    """Runs every dtype, layout and interleaved buffer type the ttnn.fold docstring lists, and a ROW_MAJOR input with
    each padding form. The output takes the input's memory configuration by default, and the other interleaved buffer
    type through override_memory_config together with collapse_output.

    A BFLOAT16 TILE input leaves out NaN, as the docstring requires, and compares zeros by value. A BFLOAT8_B input is
    compared with its own block-float values, which its BFLOAT16 output holds exactly.
    """
    check_readable(dtype, layout, buffer_type)
    memory_config = interleaved_memory_config(buffer_type)
    other_buffer_type = ttnn.BufferType.L1 if buffer_type == ttnn.BufferType.DRAM else ttnn.BufferType.DRAM
    bfloat16_tile = dtype == ttnn.bfloat16 and layout == ttnn.TILE_LAYOUT
    cases = _DOCSTRING_CASES + (_DOCSTRING_PADDED_CASES if layout == ttnn.ROW_MAJOR_LAYOUT else [])
    for shape, stride_h, stride_w, padding in cases:
        input_tensor = ttnn.from_torch(
            full_range_input(shape, dtype, nan=not bfloat16_tile),
            dtype=dtype,
            layout=layout,
            device=device,
            memory_config=memory_config,
        )
        for collapse_output, override in ((False, None), (True, interleaved_memory_config(other_buffer_type))):
            output = ttnn.fold(
                input_tensor,
                stride_h,
                stride_w,
                padding=padding,
                override_memory_config=override,
                collapse_output=collapse_output,
            )

            assert output.layout == ttnn.ROW_MAJOR_LAYOUT
            assert output.dtype == _fold_output_dtype(dtype)
            assert output.memory_config() == (memory_config if override is None else override)
            expected = _fold_reference(ttnn.to_torch(input_tensor), stride_h, stride_w, padding, collapse_output)
            assert_exact(expected, ttnn.to_torch(output), signed_zeros=not bfloat16_tile)


def _height_sharded_l1(rows, width, num_cores, device):
    """HEIGHT_SHARDED L1 memory configuration of a tensor of rows x width elements, over num_cores cores."""
    grid = ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size(), True)
    shard_spec = ttnn.ShardSpec(grid, [rows // num_cores, width], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.fold))
def test_fold_docstring_sharded_inputs(dtype, layout, device):
    """Runs every dtype and layout the ttnn.fold docstring lists on a HEIGHT_SHARDED L1 input with a shard width of C.
    A TILE input has W=32, so its rows hold no tile padding. A ROW_MAJOR input, with C=8 (16 or 32 bytes), also takes
    H and W padding, and C padding to 16. The output stays HEIGHT_SHARDED in L1."""
    check_readable(dtype, layout)
    bfloat16_tile = dtype == ttnn.bfloat16 and layout == ttnn.TILE_LAYOUT
    if layout == ttnn.TILE_LAYOUT:
        (n, h, w, c), num_cores, paddings = (1, 4, 32, 32), 2, [[0, 0]]
    else:
        (n, h, w, c), num_cores, paddings = (2, 8, 8, 8), 4, [[0, 0], [1, 1], [0, 0, 0, 0, 2, 6]]
    input_tensor = ttnn.from_torch(
        full_range_input((n, h, w, c), dtype, nan=not bfloat16_tile),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=_height_sharded_l1(n * h * w, c, num_cores, device),
    )
    for padding in paddings:
        output = ttnn.fold(input_tensor, 2, 2, padding=padding)

        assert output.layout == ttnn.ROW_MAJOR_LAYOUT
        assert output.dtype == _fold_output_dtype(dtype)
        assert output.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        assert output.memory_config().buffer_type == ttnn.BufferType.L1
        expected = _fold_reference(ttnn.to_torch(input_tensor), 2, 2, padding)
        assert_exact(expected, ttnn.to_torch(output), signed_zeros=not bfloat16_tile)


def _docstring_override_shard_layouts():
    """Shard layouts the ttnn.fold docstring lists for override_memory_config."""
    try:
        line = next(line for line in ttnn.fold.__doc__.splitlines() if "override_memory_config (" in line)
        names = re.findall(r"\b((?:HEIGHT|WIDTH|BLOCK)_SHARDED)\b", line.split("Defaults to")[0])
        assert names, f"no shard layouts named on: {line.strip()!r}"
        return [pytest.param(getattr(ttnn.TensorMemoryLayout, name), id=name) for name in dict.fromkeys(names)]
    except Exception as error:
        reason = UnreadableDocstring(f"could not read the override_memory_config line: {error}")
        return [pytest.param(reason, id="docstring-unreadable")]


@pytest.mark.parametrize("memory_layout", _docstring_override_shard_layouts())
def test_fold_docstring_sharded_override(memory_layout, device):
    """Folds an interleaved input into each shard layout the docstring lists for override_memory_config, in L1, with
    a shard spec and without one."""
    check_readable(memory_layout)
    x = full_range_input((1, 8, 8, 16), ttnn.bfloat16)
    input_tensor = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    # The output is [1, 4, 4, 64]: 16 rows of 64 values, split over 4 cores.
    grid = ttnn.num_cores_to_corerangeset(4, device.compute_with_storage_grid_size(), True)
    shard_shape = {
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED: [4, 64],
        ttnn.TensorMemoryLayout.WIDTH_SHARDED: [16, 16],
        ttnn.TensorMemoryLayout.BLOCK_SHARDED: [8, 32],
    }[memory_layout]
    if memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    with_spec = ttnn.MemoryConfig(
        memory_layout, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    )
    for override in (with_spec, ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1)):
        output = ttnn.fold(input_tensor, 2, 2, override_memory_config=override)

        assert output.memory_config().memory_layout == memory_layout
        assert output.memory_config().buffer_type == ttnn.BufferType.L1
        assert output.memory_config().shard_spec is not None
        assert_exact(_fold_reference(x, 2, 2), ttnn.to_torch(output))


@pytest.mark.parametrize("dtype", [ttnn.int32, ttnn.uint32, ttnn.uint8])
def test_fold_rejects_unsupported_row_major_dtype(device, dtype, expect_error):
    tt_input = ttnn.from_torch(
        torch.zeros((1, 16, 16, 32), dtype=torch.int32), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    with expect_error(RuntimeError, r"row-major output dtype \(.*\) must match the input dtype"):
        ttnn.fold(tt_input, 2, 2)


def run_fold_sharded_test(device, act_shape, stride_h, stride_w, padding, core_grid, layout=ttnn.ROW_MAJOR_LAYOUT):
    torch.manual_seed(0)

    N, H, W, C = act_shape
    shape = act_shape
    for run in range(2):
        torch_input = torch.randn(shape, dtype=torch.bfloat16)

        expected = fold_torch(torch_input, stride_h, stride_w, padding=padding)

        if core_grid is None:
            # Fit to max cores that divides total elements without padding. In case of padding cases,
            # reshard in fold will take care of it.
            total_elements = N * H * W
            for grid_x, grid_y in [(8, 8), (4, 4), (2, 2), (1, 1)]:
                n_cores = grid_x * grid_y
                if total_elements % n_cores == 0:
                    break
        else:
            # Currently, unaligned channel inputs are supported only without reshard path(Issue #29514).
            if len(padding) == 2:
                pad_top, pad_bottom = padding[0], padding[0]
                pad_left, pad_right = padding[1], padding[1]
            elif len(padding) == 4:
                pad_top, pad_bottom, pad_left, pad_right = padding
            else:
                raise ValueError(f"Padding must be a 2-tuple or 4-tuple, got {padding}")
            padded_H = H + pad_top + pad_bottom
            padded_W = W + pad_left + pad_right
            total_elements = N * padded_H * padded_W
            grid_x, grid_y = core_grid
            n_cores = grid_x * grid_y
            if total_elements % n_cores != 0:
                pytest.skip(f"total elements {total_elements} not divisible by n_cores {n_cores}")
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(grid_x - 1, grid_y - 1),
                ),
            }
        )

        shard_spec = ttnn.ShardSpec(shard_grid, [total_elements // n_cores, C], ttnn.ShardOrientation.ROW_MAJOR)

        tt_input = torch2tt_tensor(
            torch_input,
            device,
            layout,
            tt_memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec),
        )
        tt_out = ttnn.fold(tt_input, stride_h=stride_h, stride_w=stride_w, padding=list(padding))
        actual = tt2torch_tensor(tt_out)

        assert torch.equal(expected, actual), f"Expected: {expected}, Actual: {actual}"
        tt_input.deallocate()
        tt_out.deallocate()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "act_shape,stride_h,stride_w,padding, core_grid",
    [
        ((8, 224, 14, 8), 16, 1, (0, 0), None),
        ((1, 16, 16, 8), 2, 2, (0, 0), None),
        ((16, 42, 42, 64), 6, 6, (0, 0), None),
        ((1, 16, 16, 8), 2, 2, (2, 2), None),
        ((4, 64, 64, 24), 2, 2, (0, 0), None),
        ((4, 62, 62, 24), 2, 2, (1, 1), None),
        ((4, 14, 14, 256), 2, 2, (1, 1), None),
        ((1, 20, 20, 16), 4, 4, (2, 2), None),
    ],
)
def test_fold_sharded(device, act_shape, stride_h, stride_w, padding, core_grid):
    run_fold_sharded_test(device, act_shape, stride_h, stride_w, padding, core_grid)


def test_fold_tile_partial_tile_width_rejected(device, expect_error):
    """Logical W=48 padded to 64: 64%32==0 hides the partial-tile W the tile-native writer would OOB into. validate_fold must FATAL on logical_shape now."""
    shape = (1, 32, 48, 32)
    x = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_in = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    with expect_error(RuntimeError, "stride_w|logical W"):
        ttnn.fold(ttnn_in, 32, 32)


def test_fold_transpose_rejects_empty_grid(device, expect_error):
    x = torch.rand((1, 3, 32, 32), dtype=torch.bfloat16)
    in_sharded_memory_config = ttnn.create_sharded_memory_config(
        x.shape,
        core_grid=ttnn.CoreGrid(y=1, x=2),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    tt_in = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_sharded_memory_config)
    with expect_error(RuntimeError, "core_grid must contain at least one core"):
        ttnn.fold(tt_in, 2, 2, use_transpose_as_fold=True, padding=[0, 0, 0, 0, 0, 1], grid_size=ttnn.CoreRangeSet([]))
