# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal

from models.utility_functions import (
    pad_and_fold_conv_activation_for_unity_stride,
    pad_and_fold_conv_filters_for_unity_stride,
    _nearest_y,
    skip_for_wormhole_b0,
    torch2tt_tensor,
    tt2torch_tensor,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull


def fold_torch(input_tensor, stride_h, stride_w):
    N, H, W, C = input_tensor.shape

    reshaped = input_tensor.reshape(N, H // stride_h, stride_h, W // stride_w, stride_w, C)
    transposed = reshaped.permute(0, 1, 3, 2, 4, 5)
    return transposed.reshape(N, H // stride_h, W // stride_w, C * stride_h * stride_w)


def pad_and_fold_with_permute_and_reshape(activation_pyt_nchw_tensor, pad_h, pad_w, stride_h, stride_w):
    # pad
    C = _nearest_y(activation_pyt_nchw_tensor.shape[1], 4)
    activation_pyt_padded = torch.nn.functional.pad(
        activation_pyt_nchw_tensor, (pad_w, pad_w, pad_h, pad_h, 0, C - activation_pyt_nchw_tensor.shape[1])
    )
    # unpad params
    n, c, h, w = activation_pyt_padded.shape
    target_h = h // stride_h
    target_w = w // stride_w
    # pad to 256, 256
    n, c, h, w = activation_pyt_padded.shape
    pad_h = 256 - h
    pad_w = 256 - w
    activation_pyt_padded = torch.nn.functional.pad(activation_pyt_padded, (0, pad_w, 0, pad_h))
    # transpose
    n, c, h, w = activation_pyt_padded.shape
    activation_pyt_padded = torch.permute(activation_pyt_padded, (0, 1, 3, 2))
    n, c, w, h = activation_pyt_padded.shape
    # transpose
    activation_pyt_padded = torch.permute(activation_pyt_padded, (0, 2, 1, 3))
    n, w, c, h = activation_pyt_padded.shape
    # reshape
    activation_pyt_padded = torch.reshape(activation_pyt_padded, (n, w // stride_w, c * stride_w, h))
    n, w, c, h = activation_pyt_padded.shape
    # transpose
    activation_pyt_padded = torch.permute(activation_pyt_padded, (0, 1, 3, 2))
    n, w, h, c = activation_pyt_padded.shape
    # reshape
    activation_pyt_padded = torch.reshape(activation_pyt_padded, (n, w, h // stride_h, c * stride_h))
    n, w, h, c = activation_pyt_padded.shape
    # transpose
    activation_pyt_padded = torch.permute(activation_pyt_padded, (0, 2, 1, 3))
    n, h, w, c = activation_pyt_padded.shape
    # unpad
    activation_pyt_padded = activation_pyt_padded[:, :target_h, :target_w, :]
    return activation_pyt_padded


@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [224])
@pytest.mark.parametrize("w", [224])
@pytest.mark.parametrize("pad_h", [3])
@pytest.mark.parametrize("pad_w", [3])
@pytest.mark.parametrize("stride_h", [2])
@pytest.mark.parametrize("stride_w", [2])
def test_fold_with_permute_reshape_on_host(device, n, c, h, w, pad_h, pad_w, stride_h, stride_w):
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = pad_and_fold_conv_activation_for_unity_stride(
        torch_input_tensor, pad_h, pad_w, stride_h, stride_w
    )
    torch_output_tensor = torch.permute(torch_output_tensor, (0, 2, 3, 1))
    torch_output_tensor_new = pad_and_fold_with_permute_and_reshape(
        torch_input_tensor, pad_h, pad_w, stride_h, stride_w
    )
    assert_with_pcc(torch_output_tensor, torch_output_tensor_new, 1)


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
        (n - 1, target_h - 1, target_w - 1, c - 1),
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
    shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, c), ttnn.ShardOrientation.ROW_MAJOR, False)
    slice_sharded_memory_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    tt_output_tensor = ttnn.slice(
        tt_output_tensor,
        (0, 0, 0, 0),
        (n - 1, target_h - 1, target_w - 1, c - 1),
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
def test_fold_with_permute_reshape_on_device_sharded(
    device, n, c, h, w, pad_h, pad_w, stride_h, stride_w, use_program_cache
):
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
    tt_output_tensor = ttnn.fold(
        tt_input_tensor,
        stride_h,
        stride_w,
        use_transpose_as_fold=True,
        pad_c=_nearest_y(c, 4) - c,
        pad_h=pad_h,
        pad_w=pad_w,
        grid_size=(8, 8),
    )
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)
    assert_with_pcc(torch_output_tensor, tt_output_tensor, 1)


@skip_for_grayskull("Grayskull has pcc issue when transpose used untilize")
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
        pad_c=C - c,
        pad_h=0,
        pad_w=0,
    )
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)
    assert_with_pcc(torch_output_tensor, tt_output_tensor, 1)


# @skip_for_wormhole_b0()
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
    expected = expected.reshape(1, 1, -1, expected.shape[-1])

    tt_input = torch2tt_tensor(
        torch_input,
        device,
        ttnn.ROW_MAJOR_LAYOUT,
        tt_memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_out = ttnn.fold(tt_input, stride_h, stride_w)
    actual = tt2torch_tensor(tt_out)

    torch.testing.assert_allclose(actual, expected)


@skip_for_wormhole_b0()
def test_fold_sharded(device):
    torch.manual_seed(0)

    shape = (20, 230, 115, 8)
    N, H, W, C = shape
    stride_h = 2
    stride_w = 1

    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    expected = fold_torch(torch_input, stride_h, stride_w)
    expected = expected.reshape(1, 1, -1, expected.shape[-1])

    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(11, 7),
            ),
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 8),
                ttnn.CoreCoord(3, 8),
            ),
        }
    )
    n_cores = 100

    shard_spec = ttnn.ShardSpec(shard_grid, [N * H * W // n_cores, C], ttnn.ShardOrientation.ROW_MAJOR, False)

    tt_input = torch2tt_tensor(
        torch_input,
        device,
        ttnn.ROW_MAJOR_LAYOUT,
        tt_memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec),
    )
    tt_out = ttnn.fold(tt_input, stride_h, stride_w)
    actual = tt2torch_tensor(tt_out)

    torch.testing.assert_allclose(actual, expected)
