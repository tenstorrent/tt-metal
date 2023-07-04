import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")
import torch
import tt_lib as ttl
from tt_lib.utils import _nearest_32

def create_conv_act_tensor(torch_tensor, N, C, H, W):
    act_shape = [N, C, H, W]
    act_shape_channel_padded = [N, _nearest_32(C), H, W]
    A_ = ttl.tensor.Tensor(
        torch.flatten(torch_tensor).tolist(),
        act_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR).pad(act_shape_channel_padded, (0,0,0,0), 0.0)
    A_cl_host = A_.to(ttl.tensor.Layout.CHANNELS_LAST)
    return A_cl_host

def create_conv_weight_tensor(torch_tensor, K, C, R, S, in1_block_h, in1_block_w):
    weights_shape = [K,C,R,S]
    weights_channels_padded_shape = [_nearest_32(K),_nearest_32(C),R,S]
    B_ = ttl.tensor.Tensor(
        torch.flatten(torch_tensor).tolist(),
        weights_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR
    ).pad(weights_channels_padded_shape, (0,0,0,0), 0.0)
    B_tiled_host = ttl.tensor.convert_conv_weight_tensor_to_tiled_layout(B_, in1_block_h, in1_block_w)
    return B_tiled_host
