# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tt_lib.utils import _nearest_32, _nearest_y


def create_conv_act_tensor_special(torch_tensor, N, C, H, W, pad_h=0, pad_w=0, extra_pad_w_right=0):
    # Convert NCHW to NHWC shape
    torch_tensor = torch.permute(torch_tensor, (0, 2, 3, 1))
    # Padded input shape
    act_shape_height_width_channel_padded = [N, H + (2 * pad_h), W + (2 * pad_w) + extra_pad_w_right, _nearest_y(C, 4)]
    tt_tensor = ttnn.Tensor(torch_tensor, ttnn.bfloat16)
    h_start = pad_h if pad_h > 0 else 0
    w_start = pad_w if pad_w > 0 else 0
    tt_tensor = tt_tensor.pad(act_shape_height_width_channel_padded, (0, h_start, w_start, 0), 0.0)
    return tt_tensor


def create_conv_act_tensor(torch_tensor, N, C, H, W, pad_h=0, pad_w=0, extra_pad_w_right=0):
    # Convert NCHW to NHWC shape
    torch_tensor = torch.permute(torch_tensor, (0, 2, 3, 1))
    # Padded input shape
    act_shape_height_width_channel_padded = [N, H + (2 * pad_h), W + (2 * pad_w) + extra_pad_w_right, _nearest_y(C, 16)]
    tt_tensor = ttnn.Tensor(torch_tensor, ttnn.bfloat16)
    h_start = pad_h if pad_h > 0 else 0
    w_start = pad_w if pad_w > 0 else 0
    tt_tensor = tt_tensor.pad(act_shape_height_width_channel_padded, (0, h_start, w_start, 0), 0.0)
    return tt_tensor


def create_conv_bias_tensor(torch_tensor, N, K, padded_K, pad=0):
    # Padded input shape
    bias_shape = [N, 1, 1, K]
    bias_padded_shape = [N, 1, 1, padded_K]
    # bias_shape_padded = [N, 1, 1, _nearest_y(C, 16)]
    tt_tensor = ttnn.Tensor(torch.flatten(torch_tensor).tolist(), bias_shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT).pad(
        bias_padded_shape, (0, 0, 0, 0), 0.0
    )
    tt_tensor = tt_tensor.pad_to_tile(pad).to(ttnn.TILE_LAYOUT)
    print(f"tt_tensor shape: {tt_tensor.get_legacy_shape()}")
    return tt_tensor


def create_conv_weight_tensor(torch_tensor, K, C, R, S, in1_block_h, in1_block_w):
    weights_shape = [K, C, R, S]
    weights_channels_padded_shape = [_nearest_32(K), _nearest_y(C, 16), R, S]
    B_ = ttnn.Tensor(torch.flatten(torch_tensor).tolist(), weights_shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT).pad(
        weights_channels_padded_shape, (0, 0, 0, 0), 0.0
    )
    B_tiled_host = ttnn.operations.conv2d.convert_conv_weight_tensor_to_tiled_layout(B_, in1_block_h, in1_block_w)
    return B_tiled_host


def create_conv_weight_tensor_special_special(torch_tensor, K, C, R, S, in1_block_h, in1_block_w, padded_S=0):
    if padded_S == 0:
        padded_S = S
    else:
        assert padded_S > S
    weights_shape = [K, C, R, S]
    weights_channels_padded_shape = [_nearest_32(K), _nearest_y(C, 4), R, padded_S]
    B_ = ttnn.Tensor(torch.flatten(torch_tensor).tolist(), weights_shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT).pad(
        weights_channels_padded_shape, (0, 0, 0, 0), 0.0
    )
    B_tiled_host = ttnn.operations.conv2d.convert_conv_weight_tensor_to_special_padding_tiled_layout(
        B_, in1_block_h, in1_block_w
    )
    return B_tiled_host


def create_conv_weight_tensor_special_padding(torch_tensor, K, C, R, S, in1_block_h, in1_block_w, padded_S=0):
    if padded_S == 0:
        padded_S = S
    else:
        assert padded_S > S
    weights_shape = [K, C, R, S]
    weights_channels_padded_shape = [_nearest_32(K), _nearest_y(C, 16), R, padded_S]
    B_ = ttnn.Tensor(torch.flatten(torch_tensor).tolist(), weights_shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT).pad(
        weights_channels_padded_shape, (0, 0, 0, 0), 0.0
    )
    B_tiled_host = ttnn.operations.conv2d.convert_conv_weight_tensor_to_special_padding_tiled_layout(
        B_, in1_block_h, in1_block_w
    )
    return B_tiled_host
