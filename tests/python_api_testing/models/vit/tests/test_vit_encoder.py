from transformers import ViTForImageClassification as HF_ViTForImageClassication
from transformers import AutoImageProcessor as HF_AutoImageProcessor
from loguru import logger
import torch

from models.vit.tt.modeling_vit import TtViTEncoder
import tt_lib

from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)

from tests.python_api_testing.models.utility_functions_new import comp_pcc, comp_allclose_and_pcc


def test_vit_encoder(hf_cat_image_sample_input, pcc=0.92):
    image = hf_cat_image_sample_input

    head_mask = 12 * [None]
    output_attentions = False
    output_hidden_states = False
    return_dict = True

    with torch.no_grad():
        HF_model = HF_ViTForImageClassication.from_pretrained(
            "google/vit-base-patch16-224"
        )
        image_processor = HF_AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        inputs = image_processor(image, return_tensors="pt")["pixel_values"]

        state_dict = HF_model.state_dict()
        embedding_output = HF_model.vit.embeddings(inputs, None, None)

        reference = HF_model.vit.encoder
        config = HF_model.config

        HF_output = reference(
            embedding_output,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
        )[0]

        # Initialize the device
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)


        tt_embedding_output = torch_to_tt_tensor_rm(
            embedding_output, device, put_on_device=False
        )
        tt_layer = TtViTEncoder(
            config, base_address="vit.encoder", state_dict=state_dict, device=device
        )

        tt_output = tt_layer(
            tt_embedding_output,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
        )[0]
        tt_output = tt_to_torch_tensor(tt_output).squeeze(0)
        pcc_passing, _ = comp_pcc(HF_output, tt_output, pcc)
        _, pcc_output = comp_allclose_and_pcc(HF_output, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        tt_lib.device.CloseDevice(device)
        assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."
