import pytest

from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

from libs import tt_lib as ttl
from libs.tt_lib.utils import tilize_to_list, tilize, untilize, channels_last, _nearest_32, convert_weights_2d_matrix

from python_api_testing.models.utility_functions import print_diff_argmax, is_close
from python_api_testing.sweep_tests.comparison_funcs import comp_pcc
import torch


def run_tilize_conv3x3s1_act_test(device, K, C, H, W):
    a_activation_shape = [1, C, H, W]
    b_weights_shape = [K, C, 3, 3]
    host = ttl.device.GetHost()

    A_pyt = torch.randn(a_activation_shape, dtype=torch.bfloat16).float()
    A_ = ttl.tensor.Tensor(
        torch.flatten(A_pyt).tolist(),
        a_activation_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR)
    A_cl = A_.to(ttl.tensor.Layout.CHANNELS_LAST)
    A = A_cl.to(device, ttl.tensor.MemoryConfig(False, 0))

    # Tilize conv activation on device
    A_tiled = ttl.tensor.tilize_conv_activation(A, False)

    # Prepare weights
    B_pyt = torch.randn(b_weights_shape, dtype=torch.bfloat16).float()

    B_ = ttl.tensor.Tensor(
        torch.flatten(B_pyt).tolist(),
        b_weights_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR
    )
    B_tiled_ = ttl.tensor.convert_conv_weight_tensor_to_tiled_layout(B_)
    B_tiled = B_tiled_.to(device)
    assert A_tiled.shape()[3] == B_tiled.shape()[2]
    # Run matmul on device
    C_t = ttl.tensor.bmm(A_tiled, B_tiled)
    OH = H - 2
    OW = W - 2
    matmul_output_shape_t = [1, 1, _nearest_32(OH * OW), K]
    assert C_t.shape() == matmul_output_shape_t
    tt_host_rm = np.array(C_t.to(host).data(), dtype=float)
    pyt_got_back = torch.Tensor(tt_host_rm).reshape(matmul_output_shape_t)
    # untilize and remove padding
    C_ut = untilize(pyt_got_back)[:, :, 0 : (OH * OW), :]
    # Convert matmul output layout to conv output layout
    C_tr = torch.transpose(C_ut, 2, 3)
    assert list(C_tr.shape) == [1, 1, K, (OH * OW)]
    C_result = C_tr.reshape([1, K, OH, OW])

    # Calculate conv result with golden result. Run Pytorch conv
    C_golden = torch.nn.functional.conv2d(A_pyt, B_pyt)
    passing_pcc, output_pcc = comp_pcc(C_golden, C_result, 0.99)
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    assert passing_pcc


def test_run_tilize_conv3x3s1_act_test():
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    run_tilize_conv3x3s1_act_test(device, 32, 32, 5, 5)
    ttl.device.CloseDevice(device)
