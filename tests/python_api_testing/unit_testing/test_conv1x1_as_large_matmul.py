import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

from libs import tt_lib as ttl
from python_api_testing.models.utility_functions import untilize, tilize_to_list, channels_last, convert_weights_2d_matrix, print_diff_argmax, pad_weight, is_close
from python_api_testing.sweep_tests.comparison_funcs import comp_pcc
import torch

@pytest.mark.parametrize(
    "K, C, H, W, untilize_out, use_single_bank_reader, matmul_blocked",
    (
        (128, 128, 32, 16, False, False, True), # multi bank + multi blocks
        (128, 128, 16, 16, False, True, False), # single bank + single block
        (128, 128, 16, 16, False, False, False), # multi bank + single block
    ),
)
def test_run_1x1conv_as_large_matmul(K, C, H, W, untilize_out, use_single_bank_reader, matmul_blocked):
    #torch.manual_seed(0)
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    a_activation_shape = [1,C,H,W]
    b_weights_shape = [K,C,1,1]
    mm_output_shape = [1,1,H*W,K]

    A_pyt = torch.randn(a_activation_shape, dtype=torch.bfloat16).float()
    A_cl = channels_last(A_pyt)
    A = ttl.tensor.Tensor(
        torch.flatten(A_cl).tolist(),
        a_activation_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.CHANNELS_LAST,
        device,
        ttl.tensor.MemoryConfig(False, 0) if use_single_bank_reader else ttl.tensor.MemoryConfig(True, -1)
        )

    # Prepare weights
    B_pyt = torch.randn(b_weights_shape, dtype=torch.bfloat16).float()
    B_matrix = convert_weights_2d_matrix(B_pyt, b_weights_shape)
    assert(B_matrix.shape[0] == 1 and B_matrix.shape[1] == 1)
    assert(B_matrix.shape[2] == C and B_matrix.shape[3] == K)
    B_t = ttl.tensor.Tensor(
        tilize_to_list(B_matrix),
        B_matrix.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device,
        ttl.tensor.MemoryConfig(False, 0) if use_single_bank_reader else ttl.tensor.MemoryConfig(True, -1)
        )

    # Run TT metal OP
    if matmul_blocked:
        out = ttl.tensor.conv_as_large_bmm_single_core(A, B_t, untilize_out)
    else:
        out = ttl.tensor.conv_as_large_bmm_single_core_single_block(A, B_t, untilize_out, use_single_bank_reader)

    assert(out.shape() == mm_output_shape)
    out_pytorch = torch.tensor(out.to(host).data()).reshape(mm_output_shape)
    if not untilize_out:
        out_pytorch = untilize(out_pytorch)
    OH = H
    OW = W
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
    ttl.device.CloseDevice(device)
