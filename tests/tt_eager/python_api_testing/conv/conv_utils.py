# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tt_lib.utils import _nearest_32


def create_conv_act_tensor(torch_tensor, N, C, H, W):
    torch_tensor = torch.permute(torch_tensor, (0, 2, 3, 1))
    act_shape_channel_padded = [N, H, W, _nearest_32(C)]
    tt_tensor = ttnn.Tensor(torch_tensor, ttnn.bfloat16)
    tt_tensor = tt_tensor.pad(act_shape_channel_padded, (0, 0, 0, 0), 0.0)
    return tt_tensor


def create_conv_weight_tensor(torch_tensor, K, C, R, S, in1_block_h, in1_block_w):
    weights_shape = [K, C, R, S]
    weights_channels_padded_shape = [_nearest_32(K), _nearest_32(C), R, S]
    B_ = ttnn.Tensor(torch.flatten(torch_tensor).tolist(), weights_shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT).pad(
        weights_channels_padded_shape, (0, 0, 0, 0), 0.0
    )
    B_tiled_host = ttnn.experimental.tensor.convert_conv_weight_tensor_to_tiled_layout(B_, in1_block_h, in1_block_w)
    return B_tiled_host
