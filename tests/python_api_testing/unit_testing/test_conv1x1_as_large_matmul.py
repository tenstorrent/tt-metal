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

@pytest.mark.parametrize(
    "K, C, H, W",
    (
        (64, 64, 32, 16),
        (64, 64, 10, 10),
    ),
)
def test_run_1x1conv_as_large_matmul(K, C, H, W):
    #torch.manual_seed(0)
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    a_activation_shape = [1,C,H,W]
    b_weights_shape = [K,C,1,1]
    OH = H
    OW = W
    mm_output_shape = [1,1,_nearest_32(OH*OW),K]

    A_pyt = torch.randn(a_activation_shape, dtype=torch.bfloat16).float()
    A_ = ttl.tensor.Tensor(
        torch.flatten(A_pyt).tolist(),
        a_activation_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR)
    A_cl = A_.to(ttl.tensor.Layout.CHANNELS_LAST)
    A = A_cl.to(device)

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

    # Run TT metal OP
    out = ttl.tensor.conv_as_large_bmm_single_core(A, B_tiled)
    assert(out.shape() == mm_output_shape)
    out_pytorch = torch.tensor(out.to(host).data()).reshape(mm_output_shape)
    ttl.device.CloseDevice(device)
    # remove padding
    out_pytorch = out_pytorch[:, :, 0 : (OH * OW), :]

    # Convert matmul output layout to conv output layout
    out_tr = torch.transpose(out_pytorch, 2, 3)
    assert(list(out_tr.shape) == [1,1,K,(OH*OW)])
    out_result = out_tr.reshape([1,K,OH,OW])

    # Calculate conv result with golden result. Run Pytorch conv
    out_golden = torch.nn.functional.conv2d(A_pyt, B_pyt)
    assert(out_result.shape == out_golden.shape)
    passing_pcc, output_pcc = comp_pcc(out_golden, out_result, 0.99)
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    assert passing_pcc
