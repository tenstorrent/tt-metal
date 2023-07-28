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
from transformers import DeiTForImageClassification, AutoImageProcessor, DeiTModel
from loguru import logger

import tt_lib
from utility_functions_new import torch_to_tt_tensor_rm, tt_to_torch_tensor, comp_pcc, comp_allclose_and_pcc

from deit_config import DeiTConfig
from deit_model import TtDeiTModel


def test_deit_model_inference(hf_cat_image_sample_input, pcc = 0.95):

    head_mask = None
    output_attentions = None
    output_hidden_states = None
    return_dict = True
    bool_masked_pos = None

    with torch.no_grad():

        # setup pytorch model
        model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")
        model.eval()
        state_dict = model.state_dict()
        base_address = 'deit'

        # synthesize the input
        image = hf_cat_image_sample_input
        image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
        input_image = image_processor(images=image, return_tensors="pt")
        input_image = input_image['pixel_values']

        config = model.config
        torch_model = model.deit

        torch_output = torch_model(input_image,
                                   bool_masked_pos,
                                   head_mask,
                                   output_attentions,
                                   output_hidden_states,
                                   return_dict)[0]

        # Initialize the device
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)


        # setup tt model
        tt_image = torch_to_tt_tensor_rm(input_image, device, put_on_device=False)
        tt_model = TtDeiTModel(config,
                                device,
                                state_dict=state_dict,
                                base_address=base_address,
                                add_pooling_layer= False,
                                use_mask_token=False)

        tt_model.get_head_mask = model.get_head_mask

        tt_output = tt_model(tt_image,
                            bool_masked_pos,
                            head_mask,
                            output_attentions,
                            output_hidden_states,
                            return_dict)[0]

        tt_output = tt_to_torch_tensor(tt_output).squeeze(0)

        pcc_passing, _ = comp_pcc(torch_output, tt_output, pcc)
        _, pcc_output = comp_allclose_and_pcc(torch_output, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        tt_lib.device.CloseDevice(device)
        assert(pcc_passing), f"Failed! Low pcc: {pcc}."
