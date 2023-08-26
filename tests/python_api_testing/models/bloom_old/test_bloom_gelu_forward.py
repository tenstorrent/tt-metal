from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import tt_lib as ttm

from transformers import BloomForCausalLM
from tt_models.utility_functions import print_diff_argmax
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from loguru import logger

import python_api_testing.models.bloom_old.bloom_utils as bloom_utils
import python_api_testing.models.bloom_old.bloom_gelu_forward as bloom_gelu_forward

def run_bloom_gelu_forward_test(device):

    # Prepare input
    torch.manual_seed(0)
    test_in = torch.rand(1, 1, 256, 256) / 256

    pt_out = bloom_gelu_forward.bloom_gelu_forward(test_in)

    tt_test_in = bloom_utils.torch2tt_tensor(test_in, device)

    tt_out = bloom_gelu_forward.tt_bloom_gelu_forward(tt_test_in, device)

    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    print_diff_argmax(pt_out, tt_out_converted)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.98)

    print(comp_allclose(pt_out, tt_out_converted))
    print(pcc_message)

    if does_pass:
        logger.info("bloom_gelu_forward: Passed!")
    else:
        logger.warning("bloom_gelu_forward: Failed!")


    assert does_pass


def test_bloom_gelu_forward():
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_bloom_gelu_forward_test(device)
    ttm.device.CloseDevice(device)


if __name__ == "__main__":
    test_bloom_gelu_forward()
