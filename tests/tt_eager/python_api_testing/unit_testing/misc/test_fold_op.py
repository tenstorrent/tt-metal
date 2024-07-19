# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

import tt_lib as ttl
from tt_lib import tensor as tt

from models.utility_functions import skip_for_wormhole_b0, torch2tt_tensor, tt2torch_tensor

from tt_lib.utils import (
    tilize_to_list,
    tilize,
    untilize,
    _nearest_32,
    _nearest_y,
    convert_weights_2d_matrix,
)
from models.utility_functions import (
    pad_and_fold_conv_activation_for_unity_stride,
    pad_and_fold_conv_filters_for_unity_stride,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


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
    activation_pyt_padded = ttl.tensor.transpose(activation_pyt_padded, 2, 3, ttnn.L1_MEMORY_CONFIG)
    # transpose
    activation_pyt_padded = ttl.tensor.transpose(activation_pyt_padded, 1, 2, ttnn.L1_MEMORY_CONFIG)
    # reshape
    n, w, c, h = activation_pyt_padded.shape
    activation_pyt_padded = ttl.tensor.reshape(
        activation_pyt_padded, n, w // stride_w, c * stride_w, h, output_mem_config=ttnn.L1_MEMORY_CONFIG
    )
    # transpose
    activation_pyt_padded = ttl.tensor.transpose(activation_pyt_padded, 2, 3, ttnn.L1_MEMORY_CONFIG)
    # reshape
    n, w, h, c = activation_pyt_padded.shape
    activation_pyt_padded = ttl.tensor.reshape(
        activation_pyt_padded, n, w, h // stride_h, c * stride_h, output_mem_config=ttnn.L1_MEMORY_CONFIG
    )
    # transpose
    activation_pyt_padded = ttl.tensor.transpose(activation_pyt_padded, 1, 2, ttnn.L1_MEMORY_CONFIG)
    # slice
    n, h, w, c = activation_pyt_padded.shape
    activation_pyt_padded = ttnn.slice(
        activation_pyt_padded,
        (0, 0, 0, 0),
        (n - 1, target_h - 1, target_w - 1, c - 1),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    return activation_pyt_padded


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
    tt_output_tensor = pad_and_fold_with_permute_and_reshape_on_device(
        device, torch_input_tensor, pad_h, pad_w, stride_h, stride_w
    )
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)
    assert_with_pcc(torch_output_tensor, tt_output_tensor, 1)


@skip_for_wormhole_b0()
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
        tt.Layout.ROW_MAJOR,
        tt_memory_config=tt.MemoryConfig(tt.TensorMemoryLayout.INTERLEAVED, tt.BufferType.L1),
    )

    tt_out = tt.fold(tt_input, stride_h, stride_w)
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

    shard_grid = tt.CoreRangeSet(
        {
            tt.CoreRange(
                tt.CoreCoord(0, 0),
                tt.CoreCoord(11, 7),
            ),
            tt.CoreRange(
                tt.CoreCoord(0, 8),
                tt.CoreCoord(3, 8),
            ),
        }
    )
    n_cores = 100

    shard_spec = tt.ShardSpec(shard_grid, [N * H * W // n_cores, C], tt.ShardOrientation.ROW_MAJOR, False)

    tt_input = torch2tt_tensor(
        torch_input,
        device,
        tt.Layout.ROW_MAJOR,
        tt_memory_config=tt.MemoryConfig(tt.TensorMemoryLayout.HEIGHT_SHARDED, tt.BufferType.L1, shard_spec),
    )
    tt_out = tt.fold(tt_input, stride_h, stride_w)
    actual = tt2torch_tensor(tt_out)

    torch.testing.assert_allclose(actual, expected)
