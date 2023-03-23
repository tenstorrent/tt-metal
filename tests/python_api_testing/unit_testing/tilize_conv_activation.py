from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import numpy as np

from pymetal import ttlib as ttl
from pymetal.ttlib.utils import tilize_to_list, tilize, untilize, channels_last, _nearest_32, convert_weights_2d_matrix, convert_act_2d_matrix

import torch

def run_tilize_conv3x3s1_act_test (C, H, W):
    a_activation_shape = [1,C,H,W]

    A_pyt = torch.randn(a_activation_shape, dtype=torch.bfloat16).float()
    A_cl = channels_last(A_pyt)
    A = ttl.tensor.Tensor(
        torch.flatten(A_cl).tolist(),
        a_activation_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.CHANNELS_LAST,
        device,
        ttl.tensor.MemoryConfig(False, 0)
    )
    # Tilize conv activation on device
    A_t = ttl.tensor.tilize_conv_activation(A)
    OH = H - 2
    OW = W - 2
    output_shape = [1,1,_nearest_32(OH*OW),C*9]
    pyt_got_back = np.array(A_t.to(host).data(), dtype=float).reshape(output_shape)
    print("Pytorch tensor got back shape - " + str(pyt_got_back.shape))
    # untilize and remove padding
    A_ut = untilize(pyt_got_back)[:,:,0:(OH*OW),:]
    A_golden = convert_act_2d_matrix(A_pyt, 3, 3, 1, 1, 0, 0)
    assert(A_ut.shape == A_golden.shape)
    print(abs(A_golden - A_ut).max())
    assert (abs(A_golden - A_ut) < 0.02).all(), "Max abs difference for tilize can be 0.02 due to bfloat conversions"

if __name__ == "__main__":
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_tilize_conv3x3s1_act_test(32, 5, 5)
    ttl.device.CloseDevice(device)
