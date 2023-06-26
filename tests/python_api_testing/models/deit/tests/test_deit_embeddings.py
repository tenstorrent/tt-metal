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
from transformers import AutoImageProcessor,DeiTModel
from datasets import load_dataset
from loguru import logger

import tt_lib
from utility_functions_new import torch_to_tt_tensor_rm, tt_to_torch_tensor, comp_pcc, comp_allclose_and_pcc

from deit_config import DeiTConfig
from deit_embeddings import DeiTEmbeddings


def test_deit_embeddings_inference(pcc=0.99):

    # setup pytorch model
    model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model.eval()
    state_dict = model.state_dict()
    base_address= 'embeddings'
    torch_embeddings = model.embeddings
    use_mask_token = False
    bool_masked_pos = None
    head_mask = None

    #real input
    image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    input_image = image_processor(images=image, return_tensors="pt")
    input_image = input_image['pixel_values']

    torch_output = torch_embeddings(input_image, bool_masked_pos)

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    # setup tt model
    tt_embeddings = DeiTEmbeddings(DeiTConfig(),
                                    base_address,
                                    state_dict,
                                    use_mask_token)

    tt_output = tt_embeddings(input_image, bool_masked_pos)

    pcc_passing, _ = comp_pcc(torch_output, tt_output, pcc)
    _, pcc_output = comp_allclose_and_pcc(torch_output, tt_output, pcc)
    logger.info(f"Output {pcc_output}")
    assert(pcc_passing), f"Failed! Low pcc: {pcc}."
