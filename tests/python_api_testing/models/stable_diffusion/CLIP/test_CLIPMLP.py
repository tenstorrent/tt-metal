import numpy as np
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

from loguru import logger

import torch.nn as nn
import torch

from libs import tt_lib as ttl
from transformers import CLIPModel
from utility_functions import print_diff_argmax, torch_to_tt_tensor, tt_to_torch_tensor, print_corr_coef
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc




def run_test_clip_mlp_inference(device, host):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float32)
    model.eval()
    state_dict = model.state_dict()
    config = model.config.text_config

    hidden_size = config.hidden_size
    intermediate_size= config.intermediate_size
    input_shape = [1, 2, 32, hidden_size]
    input = torch.randn(input_shape)
    torch_mlp = model.text_model.encoder.layers[10].mlp

    torch_out = torch_mlp(input)

    tt_input = torch_to_tt_tensor(input, device)

    tt_mlp = TtCLIPMLP(device, config=config, state_dict=state_dict)

    tt_out = tt_mlp(tt_input)
    tt_out = tt_to_torch_tensor(tt_out, host)

    print_diff_argmax(tt_out, torch_out)

    print(comp_allclose_and_pcc(torch_out, tt_out))
    does_pass, pcc_message = comp_pcc(torch_out, tt_out, 0.99)

    print(comp_allclose(torch_out, tt_out))
    print(pcc_message)

    if does_pass:
        logger.info("test_CLIPMLP_inference Passed!")
    else:
        logger.warning("test_CLIPMLP_inference Failed!")
    assert does_pass


def test_CLIPMLP():
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_test_clip_mlp_inference(device, host)
    ttl.device.CloseDevice(device)
