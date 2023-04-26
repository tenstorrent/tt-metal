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
from utility_functions import print_diff_argmax
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from loguru import logger

import python_api_testing.models.bloom.bloom_utils as bloom_utils
import python_api_testing.models.bloom.baddbmm as baddbmm

def run_baddbmm_test(device):

    # Prepare input
    torch.manual_seed(0)
    input = torch.rand(32, 64, 64)
    batch1 = torch.rand(32, 64, 32)
    batch2 = torch.rand(32, 32, 64)

    pt_out = torch.baddbmm(input, batch1, batch2)
    pt_out_size = list(pt_out.shape)

    while len(pt_out_size) < 4:
        pt_out_size.insert(0, 1)

    pt_out = torch.reshape(pt_out, pt_out_size)

    tt_out = baddbmm.tt_baddbmm(device, input, batch1, batch2)

    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    print_diff_argmax(pt_out, tt_out_converted)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.64)

    print(comp_allclose(pt_out, tt_out_converted))
    print(pcc_message)

    if does_pass:
        logger.info("baddbmm: Passed!")
    else:
        logger.warning("baddbmm: Failed!")


    assert does_pass


def test_baddbmm():
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_baddbmm_test(device)
    ttm.device.CloseDevice(device)
