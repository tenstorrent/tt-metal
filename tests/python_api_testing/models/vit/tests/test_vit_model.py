from transformers import ViTForImageClassification as HF_ViTForImageClassication
from loguru import logger
import torch

import tt_lib
from tt_models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)

from tt_models.utility_functions import comp_pcc, comp_allclose_and_pcc
from tt_models.vit.tt.modeling_vit import TtViTModel


def test_vit_model(imagenet_sample_input, pcc=0.95):
    image = imagenet_sample_input
    head_mask = None
    output_attentions = None
    output_hidden_states = None
    interpolate_pos_encoding = None
    return_dict = None
    with torch.no_grad():
        HF_model = HF_ViTForImageClassication.from_pretrained(
            "google/vit-base-patch16-224"
        )

        state_dict = HF_model.state_dict()

        reference = HF_model.vit

        config = HF_model.config
        HF_output = reference(
            image,
            head_mask,
            output_attentions,
            output_hidden_states,
            interpolate_pos_encoding,
            return_dict,
        )[0]

        # Initialize the device
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)


        tt_image = torch_to_tt_tensor_rm(image, device, put_on_device=False)
        tt_layer = TtViTModel(
            config,
            add_pooling_layer=False,
            base_address="vit",
            state_dict=state_dict,
            device=device,
        )
        tt_layer.get_head_mask = reference.get_head_mask
        tt_output = tt_layer(
            tt_image,
            head_mask,
            output_attentions,
            output_hidden_states,
            interpolate_pos_encoding,
            return_dict,
        )[0]
        tt_output = tt_to_torch_tensor(tt_output).squeeze(0)
        pcc_passing, _ = comp_pcc(HF_output, tt_output, pcc)
        _, pcc_output = comp_allclose_and_pcc(HF_output, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."
