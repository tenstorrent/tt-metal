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
from transformers import AutoImageProcessor, DeiTForImageClassification
from loguru import logger

import tt_lib
from utility_functions_new import torch_to_tt_tensor_rm, tt_to_torch_tensor, comp_pcc, comp_allclose_and_pcc

from deit_config import DeiTConfig
from deit_for_image_classification import TtDeiTForImageClassification


def test_deit_for_image_classification_inference(hf_cat_image_sample_input, pcc=0.95):

    with torch.no_grad():
        image = hf_cat_image_sample_input

        #real input
        image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
        inputs = image_processor(images=image, return_tensors="pt")

        torch_model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")
        torch_model.eval()
        state_dict = torch_model.state_dict()
        config = torch_model.config

        torch_output = torch_model(**inputs).logits

        # Initialize the device
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)

        tt_inputs = torch_to_tt_tensor_rm(inputs["pixel_values"], device, put_on_device=False)
        tt_model = TtDeiTForImageClassification(config,
                                                device=device,
                                                state_dict=state_dict,
                                                base_address="")

        tt_model.deit.get_head_mask = torch_model.deit.get_head_mask
        tt_output = tt_model(tt_inputs)[0]
        tt_output = tt_to_torch_tensor(tt_output).squeeze(0)[:, 0, :]

        pcc_passing, _ = comp_pcc(torch_output, tt_output, pcc)
        _, pcc_output = comp_allclose_and_pcc(torch_output, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        tt_lib.device.CloseDevice(device)
        assert(pcc_passing), f"Failed! Low pcc: {pcc}."
