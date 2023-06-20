from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

from typing import Optional, Set, Tuple, Union
from PIL import Image
import requests

import torch
from torch import nn
from transformers import AutoImageProcessor,DeiTModel

from loguru import logger

import tt_lib
from utility_functions_new import torch_to_tt_tensor_rm, tt_to_torch_tensor, comp_pcc, comp_allclose_and_pcc

from deit_config import DeiTConfig
from deit_embeddings import TtDeiTEmbeddings


def test_deit_embeddings_inference():
    # setup pytorch model
    model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model.eval()
    state_dict = model.state_dict()

    # synthesize the input
    base_address= 'embeddings'
    torch_embeddings = model.embeddings
    use_mask_token = False
    bool_masked_pos = None
    head_mask = None
    input_shape =  torch.Size([1, 3, 224, 224])
    input_image_syn = torch.randn(input_shape)

    image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    input_image = image_processor(images=image, return_tensors="pt")
    input_image = input_image['pixel_values']

    # torch_output = torch_embeddings(input_image_syn, bool_masked_pos)
    torch_output = torch_embeddings(input_image, bool_masked_pos)

    print('\n in test torch output:', torch_output[0][0][0:10])
    print('in test torch output shape', torch_output.shape)

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    # setup tt model

    tt_embeddings = TtDeiTEmbeddings(DeiTConfig(), host, device, state_dict, base_address, use_mask_token= use_mask_token)

    # tt_input = torch_to_tt_tensor_rm(input_image_syn, device, put_on_device=False)
    tt_input = torch_to_tt_tensor_rm(input_image, device, put_on_device=False)

    tt_out = tt_embeddings(tt_input, bool_masked_pos)
    tt_output = tt_to_torch_tensor(tt_out, host).squeeze(0)
    print('\n in test tt output:', tt_output[0][0][0:10])
    print('in test tt output shape', tt_output.shape)


    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))
    tt_lib.device.CloseDevice(device)
    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
