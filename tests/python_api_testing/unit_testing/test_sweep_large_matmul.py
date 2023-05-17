import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np
import os
import yaml
import torch
from loguru import logger
from libs import tt_lib as ttl
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight, is_close
from python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from python_api_testing.conv.generate_mm_tb_using_conv_tb import generate_mm_tb_using_conv_tb

def run_large_matmul(Ha, Wa, Wb):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    torch.manual_seed(0)
    host = ttl.device.GetHost()
    a_shape = [1, 1, Ha, Wa]
    b_shape = [1, 1, Wa, Wb]

    a = torch.randn(a_shape, dtype=torch.bfloat16).float()
    b = torch.randn(b_shape, dtype=torch.bfloat16).float()

    layout_a = ttl.tensor.Layout.ROW_MAJOR


    tta = ttl.tensor.Tensor(
        a.flatten().tolist(),
        a_shape,
        ttl.tensor.DataType.BFLOAT16,
        layout_a,
        device)

    ttb = ttl.tensor.Tensor(
        tilize_to_list(b),
        b_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device)

    out = ttl.tensor.large_bmm(tta, ttb, True, True)
    out_shape = [1,1,Ha,Wb]
    out = out.to(host)
    out_pytorch = torch.tensor(out.data()).reshape(out_shape)
    ttl.device.CloseDevice(device)
    golden_pytorch = torch.matmul(a,b)
    assert(out_pytorch.shape == golden_pytorch.shape), f"Shape mismatch: actual: {out_pytorch.shape}, expected: {golden_pytorch.shape}"
    passing_pcc, output_pcc = comp_pcc(golden_pytorch, out_pytorch, 0.99)
    logger.debug("Passing=", passing_pcc)
    logger.debug("Output pcc=", output_pcc)
    assert passing_pcc


def test_run_sweep_large_matmul_test():
    # generate MM test bench using conv sweep test bench
    # Use pre-generated (modified) test list with failing params commented out
    # mm_tb_list = generate_mm_tb_using_conv_tb()
    with open(os.path.join(os.environ['TT_METAL_HOME'], 'tests/python_api_testing/conv/generated_mm_tb.yaml'), 'r') as file:
        mm_tb = yaml.safe_load(file)
    for [Ha,Wa,Wb] in mm_tb[0]["MM test params [M,K,N]"]:
        logger.debug("Testing MM with - ")
        logger.debug("   Act shape - " + str(Ha) + "," + str(Wa))
        logger.debug("   Weight shape - " + str(Wa) + "," + str(Wb))
        run_large_matmul(Ha, Wa, Wb)
    logger.info("All " + str(len(mm_tb[0]["MM test params [M,K,N]"])) + " tests passed!")
