from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import numpy as np

from pymetal import ttmetal as ttm
from pymetal.ttmetal.utils import tilize_to_list, tilize, untilize, channels_last, _nearest_32, pad_activation
from python_api_testing.models.utility_functions import print_diff_argmax, is_close
import torch

def convert_weights_2d_matrix(weights, w_shape):
    """
    :param x: Input PyTorch Tensor
    :type x: class:`torch.Tensor`
    """
    ret_shape = [1,1,w_shape[0],w_shape[1]*w_shape[2]*w_shape[3]]
    if isinstance(weights, torch.Tensor):
        ret = torch.zeros(np.prod(ret_shape))
    else:
        ret = np.zeros(np.prod(ret_shape))
    idx = 0
    for k in range(w_shape[0]):
        for r in range(w_shape[2]):
            for s in range(w_shape[3]):
                for c in range(w_shape[1]):
                    ret[idx] = weights[k][c][r][s]
                    idx+=1
    assert idx == np.prod(ret_shape)
    return ret.reshape(ret_shape).transpose(2,3)

def convert_cl_act_2d_matrix_3x3s1(activation):
    """
    :param x: Input PyTorch Tensor
    :type x: class:`torch.Tensor`
    """
    N = activation.shape[0]
    C = activation.shape[3]
    H = activation.shape[1]
    W = activation.shape[2]
    OH = H - 2 # for stride 1 and 3x3
    OW = W - 2 # for stride 1 and 3x3
    nrows = OH*OW
    ncols = C*9
    ret_shape = [1,1,nrows,ncols]
    if isinstance(activation, torch.Tensor):
        ret = torch.zeros(np.prod(ret_shape))
    else:
        ret = np.zeros(np.prod(ret_shape))
    idx = 0
    for n in range(N):
        for oh in range(OH): # for stride 1
            for ow in range (OW): # for stride 1
                for r in range(3): # for kernel width = 3
                    for s in range(3): # for kernel height = 3
                        for c in range(C):
                            ret[idx] = activation[n][oh+r][ow+s][c]
                            idx+=1
    assert idx == np.prod(ret_shape)
    return ret.reshape(ret_shape)


def run_tilize_conv3x3s1_act_test (K, C, H, W):
    a_activation_shape = [1,C,H,W]
    b_weights_shape = [K,C,3,3]

    A_pyt = torch.randn(a_activation_shape)
    A_cl = channels_last(A_pyt)
    A = ttm.tensor.Tensor(
        torch.flatten(A_cl).tolist(),
        a_activation_shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.CHANNELS_LAST,
        device,
        ttm.tensor.MemoryConfig(False, 0)
    )
    # Tilize conv activation on device
    A_t = ttm.tensor.tilize_conv_activation(A)

    # Prepare weights
    B_pyt = torch.randn(b_weights_shape)
    B_matrix = convert_weights_2d_matrix(B_pyt, b_weights_shape)

    B_t = ttm.tensor.Tensor(tilize_to_list(B_matrix), B_matrix.shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)
    assert(A_t.shape()[3] == B_t.shape()[2])
    # Run matmul on device
    C_t = ttm.tensor.bmm(A_t, B_t)
    OH = H - 2
    OW = W - 2
    matmul_output_shape_t = [1,1,_nearest_32(OH*OW), K]
    assert(C_t.shape() == matmul_output_shape_t)
    tt_host_rm = C_t.to(host).data()
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
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    run_tilize_conv3x3s1_act_test(32, 32, 4, 4)
    ttm.device.CloseDevice(device)
