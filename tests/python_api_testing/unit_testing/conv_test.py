from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

from pymetal import ttlib as ttl
from pymetal.ttlib.utils import tilize_to_list, tilize, untilize, channels_last, _nearest_32, convert_weights_2d_matrix

from python_api_testing.models.utility_functions import print_diff_argmax, is_close
import torch

def run_tilize_conv3x3s1_act_test (K, C, H, W):
    a_activation_shape = [1,C,H,W]
    b_weights_shape = [K,C,3,3]

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

    # Prepare weights
    B_pyt = torch.randn(b_weights_shape, dtype=torch.bfloat16).float()
    B_matrix = convert_weights_2d_matrix(B_pyt, b_weights_shape)

    B_t = ttl.tensor.Tensor(tilize_to_list(B_matrix), B_matrix.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)
    assert(A_t.shape()[3] == B_t.shape()[2])
    # Run matmul on device
    C_t = ttl.tensor.bmm(A_t, B_t)
    OH = H - 2
    OW = W - 2
    matmul_output_shape_t = [1,1,_nearest_32(OH*OW), K]
    assert(C_t.shape() == matmul_output_shape_t)
    tt_host_rm = np.array(C_t.to(host).data(), dtype=float)
    pyt_got_back = torch.Tensor(tt_host_rm).reshape(matmul_output_shape_t)
    # untilize and remove padding
    C_ut = untilize(pyt_got_back)[:,:,0:(OH*OW),:]
    # Convert matmul output layout to conv output layout
    C_tr = torch.transpose(C_ut, 2, 3)
    assert(list(C_tr.shape) == [1,1,K,(OH*OW)])
    C_result = C_tr.reshape([1,K,OH,OW])

    # Calculate conv result with golden result. Run Pytorch conv
    C_golden = torch.nn.functional.conv2d(A_pyt, B_pyt)
    maxmag = C_golden.abs().max().item() # % of max magnitude since that determines cancellations
    match = is_close(C_result, C_golden, 0.07, 0.07, maxmag, 0.01)
    print("Match=", match.item())

if __name__ == "__main__":
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_tilize_conv3x3s1_act_test(32, 32, 5, 5)
    ttl.device.CloseDevice(device)
