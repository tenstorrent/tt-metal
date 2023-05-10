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
import python_api_testing.models.bloom.bloom_model as bloom_model


def run_bloom_model_test(device):
    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m", torchscript=False)
    hugging_bloom_reference_model.eval()

    print(hugging_bloom_reference_model.config)

    config = hugging_bloom_reference_model.config
    #use_cache
    state_dict = hugging_bloom_reference_model.state_dict()
    base_address = "transformer"
    # hidden_size = config.hidden_size # 1024
    # n_head = config.n_head

    tt_bloom_model = bloom_model.TtBloomModel(config, state_dict, base_address, device)
    pt_bloom_model = hugging_bloom_reference_model.transformer

    # Prepare input
    torch.manual_seed(0)

    input_ids = torch.randint(0, 100, (1, 64))

    pt_out = pt_bloom_model.forward(input_ids)[0]
    print("PT finished")

    tt_out = tt_bloom_model.forward(device, input_ids)[0]
    print("TT finished")

    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)
    tt_out_converted = tt_out_converted.squeeze(0)

    print(f"pt_out shape {pt_out.shape}")
    print(f"tt_out_converted shape {tt_out_converted.shape}")

    print_diff_argmax(pt_out, tt_out_converted)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.65)

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


if __name__ == "__main__":
    test_bloom_model()
