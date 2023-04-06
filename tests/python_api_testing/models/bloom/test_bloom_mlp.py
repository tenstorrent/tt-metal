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
import python_api_testing.models.bloom.bloom_mlp as bloom_mlp

def run_bloom_mlp_test(device):

    # Prepare input
    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m", torchscript=False)
    state_dict = hugging_bloom_reference_model.state_dict()
    # Prepare input
    torch.manual_seed(0)

    test_in = torch.rand(1, 1, 4096, 1024)
    res = torch.rand(1, 1, 4096, 1024)

    tt_mlp = bloom_mlp.TtBloomMLP(hugging_bloom_reference_model, 0.0, 1024, False, device)

    tt_out =  tt_mlp.forward(test_in, res, device)

    pt_mlp = bloom_mlp.BloomMLP(hugging_bloom_reference_model.state_dict(), 0.0, 1024, False)

    pt_out = pt_mlp.forward(test_in, res)

    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    print_diff_argmax(pt_out, tt_out_converted)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.99)

    print(comp_allclose(pt_out, tt_out_converted))
    print(pcc_message)

    if does_pass:
        logger.info("bloom_mlp: Passed!")
    else:
        logger.warning("bloom_mlp: Failed!")


    assert does_pass

def test_bloom_mlp():
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_bloom_mlp_test(device)
    ttm.device.CloseDevice(device)
