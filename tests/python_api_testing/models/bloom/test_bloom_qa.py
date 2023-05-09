from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from libs import tt_lib as ttm

from transformers import BloomForQuestionAnswering
from utility_functions import print_diff_argmax
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from loguru import logger
import python_api_testing.models.bloom.bloom_qa as bloom_qa



def run_bloom_qa_inference(device):
    hugging_bloom_reference_model = BloomForQuestionAnswering.from_pretrained("bigscience/bloom-560m", torchscript=False)
    hugging_bloom_reference_model.eval()

    config = hugging_bloom_reference_model.config
    state_dict = hugging_bloom_reference_model.state_dict()

    tt_bloom_qa = bloom_qa.TtBloomForQuestionAnswering(config, state_dict, device)
    pt_bloom_qa = hugging_bloom_reference_model

    # Prepare input
    torch.manual_seed(0)
    input_ids = torch.randint(0, 100, (1, 64))

    pt_out = pt_bloom_qa.forward(input_ids)
    print("PT finished")

    tt_out = tt_bloom_qa.forward(device, input_ids)
    print("TT finished")

    pt_out = pt_out[0]
    tt_out = tt_out[0]
    tt_out = tt_out.squeeze(0)
    tt_out = tt_out.squeeze(0)

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.66)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    if does_pass:
        logger.info("bloom_qa: Passed!")
    else:
        logger.warning("bloom_qa: Failed!")

    assert does_pass


def test_bloom_qa():
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_bloom_qa_inference(device)
    ttm.device.CloseDevice(device)


if __name__ == "__main__":
    test_bloom_qa()
