"""Gemma-3-4b-it Test for Vision Transformer"""


# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.ccl import TT_CCL

from models.experimental.gemma3_4b.tt.gemma_vision_crossattention import TtGemmaTransformerVision
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "fabric_config": True}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("bsz", [1])
def test_gemma_vision(
    mesh_device,
    reset_seeds,
    bsz,
):
    pcc_required = 0.90
    dtype = ttnn.bfloat16
    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    vision_first_layer_prefix = "vision_tower.vision_model."
    vision_partial_state_dict = {
        k[len(vision_first_layer_prefix) :]: v
        for k, v in state_dict.items()
        if (k.startswith(vision_first_layer_prefix))
    }

    reference_vision_model = model_args.reference_vision_model()
    # reference_vision_model.load_state_dict(vision_partial_state_dict)

    mmp_first_layer_prefix = "multi_modal_projector."
    # mmp_partial_state_dict = {
    #     k[len(mmp_first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(mmp_first_layer_prefix))
    # }

    image_size = model_args.vision_chunk_size
    in_channels = model_args.vision_in_channels

    # model_id = "google/gemma-3-4b-it"
    # processor = AutoProcessor.from_pretrained(model_id)
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "image",
    #                 "image": "https://www.talkesport.com/wp-content/uploads/eentity-1024x574.jpg",
    #             },
    #             {"type": "text", "text": "Describe this?"},
    #         ],
    #     }
    # ]

    # inputs = processor.apply_chat_template(
    #     messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    # ).to(dtype=torch.bfloat16)

    # input_tensor = inputs["pixel_values"]

    input_tensor = torch.rand((bsz, in_channels, image_size, image_size))

    reference_mmp = model_args.reference_vision_multi_modal()
    # reference_mmp.load_state_dict(mmp_partial_state_dict)

    reference_output = get_image_features(
        reference_vision_model,
        reference_mmp,
        input_tensor,
    )

    tt_ccl = TT_CCL(mesh_device)
    test_gemma_vision = TtGemmaTransformerVision(
        mesh_device,
        tt_ccl,
        state_dict,
        state_dict_prefix="vision_tower.vision_model.",
        dtype=dtype,
        configuration=model_args,
        return_intermediate=False,
    )

    test_output = test_gemma_vision(input_tensor)

    logger.info("Checking outputs")
    out = ttnn.from_device(test_output)
    tt_output_torch = ttnn.to_torch(out)
    tt_output_torch = tt_output_torch.view(1, 256, 2560)
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    non_zero_indices = tt_output_torch.ne(0).nonzero(as_tuple=True)
    tt_output_torch = tt_output_torch[non_zero_indices]
    reference_output = reference_output[non_zero_indices]

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"


def get_image_features(vision_tower, projector, input_tensor):
    """
    Get image features from the vision tower and projector.
    """
    vision_token = vision_tower(input_tensor).last_hidden_state
    image_features = projector(vision_token)
    return image_features
