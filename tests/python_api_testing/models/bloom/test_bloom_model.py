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
import python_api_testing.models.bloom.bloom_attention as bloom_attention
import python_api_testing.models.bloom.bloom_block as bloom_block
import python_api_testing.models.bloom.bloom_model as bloom_model


def run_bloom_model_test(device):

    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m", torchscript=False)
    print(hugging_bloom_reference_model.state_dict())
    tt_bloom_model = bloom_model.TtBloomModel(device, hugging_bloom_reference_model, 1024, 32,  250880, 1024, 1e-5, 2)
    pt_bloom_model = bloom_model.BloomModel(hugging_bloom_reference_model, 1024, 32,  250880, 1024, 1e-5, 2)


    # Prepare input
    torch.manual_seed(0)

    input_ids = torch.randint(0, 100, (1, 64))

    pt_out = pt_bloom_model.forward(input_ids)

    print("PT---------------")
    print(pt_out[0].shape)
    #print(pt_out[1].shape)

    print("PT finished")

    tt_out = tt_bloom_model.forward(device, input_ids)

    print("TT finished")

    print("TT---------------")
    print(tt_out[0].shape())

    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out[0])
    tt_out_converted = tt_out_converted.squeeze(0)
    tt_out_converted = tt_out_converted.squeeze(0)

    pt_out = pt_out[0]


    print_diff_argmax(pt_out, tt_out_converted)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.70)

    print(comp_allclose(pt_out, tt_out_converted))
    print(pcc_message)

    if does_pass:
        logger.info("bloom_model: Passed!")
    else:
        logger.warning("bloom_model: Failed!")
    assert does_pass

def test_bloom_model():
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_bloom_model_test(device)
    ttm.device.CloseDevice(device)
