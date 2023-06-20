from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import torch
from torch import nn
from transformers import DeiTForImageClassification, AutoImageProcessor,DeiTModel
from PIL import Image
import requests
from loguru import logger

import tt_lib
from utility_functions_new import torch_to_tt_tensor_rm, tt_to_torch_tensor, comp_pcc, comp_allclose_and_pcc

from deit_config import DeiTConfig
from deit_model import TtDeiTModel


def test_deit_model_inference():
    # image = imagenet_sample_input
    head_mask = None
    output_attentions = None
    output_hidden_states = None
    interpolate_pos_encoding = None
    return_dict = None
    # setup pytorch model
    model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model.eval()
    state_dict = model.state_dict()
    base_address = 'deit'

    # synthesize the input
    image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    input_image = image_processor(images=image, return_tensors="pt")
    input_image = input_image['pixel_values']

    config = model.config

    torch_out = model(input_image, head_mask, output_attentions, output_hidden_states, interpolate_pos_encoding, return_dict)[0]
    print('\ntorch out:',torch_out[0][:10])

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    # setup tt model
    tt_image = torch_to_tt_tensor_rm(input_image, device, put_on_device=False)
    tt_model = TtDeiTModel(config, host, device, state_dict=state_dict, base_address=base_address, add_pooling_layer= True, use_mask_token=False)
    tt_model.get_head_mask = model.get_head_mask

    tt_out = tt_output(tt_image, bool_masked_pos, head_mask, output_attentions, output_hidden_states)
    tt_out = tt_to_torch_tensor(tt_out, host)

    passing = comp_pcc(torch_out, tt_out)
    logger.info(comp_allclose_and_pcc(tt_out, torch_out))
    tt_lib.device.CloseDevice(device)
    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
