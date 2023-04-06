from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from libs import tt_lib as ttm

from transformers import BloomForCausalLM
from utility_functions import print_diff_argmax, comp_allclose, comp_pcc

from loguru import logger

import python_api_testing.models.bloom.bloom_utils as bloom_utils
import python_api_testing.models.bloom.dropout_add as dropout_add

def run_dropout_add_test(device):

    # Prepare input
    torch.manual_seed(0)
    test_in = torch.rand(1, 1, 64, 64)
    res = torch.rand(1, 1, 64, 64)

    pt_out = dropout_add.dropout_add(test_in, res, 0.3, False)

    tt_out =  dropout_add.tt_dropout_add(test_in, res, 0.3, False, device)

    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    print_diff_argmax(pt_out, tt_out_converted)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.99)

    print(comp_allclose(pt_out, tt_out_converted))
    print(pcc_message)

    if does_pass:
        logger.info("dropout_add: Passed!")
    else:
        logger.warning("dropout_add: Failed!")


    assert does_pass

def test_dropout_add():
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_dropout_add_test(device)
    ttm.device.CloseDevice(device)
