# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForVision2Seq
from transformers.models.mllama.image_processing_mllama import build_aspect_ratio_mask, convert_aspect_ratios_to_ids
from transformers.models.mllama.modeling_mllama import MllamaVisionModel

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tests.multimodal.utils import load_partial_weights
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.multimodal.llama_vision_encoder import TtLlamaVisionEncoder


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_vision_encoder_inference(mesh_device, reset_seeds):
    dtype = ttnn.bfloat16
    pcc_required = 0.88

    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = "vision_model.vision_encoder."
    return_intermediate = "3,7,15,23,30"
    return_intermediate = [int(l) for l in return_intermediate.split(",")]

    model_repo_name = os.getenv("HF_MODEL")
    # config contains paramters for the whole multimodal network the subeset of vision branch is chosen instead
    config = AutoConfig.from_pretrained(model_repo_name)
    config.vision_config._attn_implementation = "sdpa"
    reference_model = MllamaVisionModel(config.vision_config)
    # partial loading of HF safetensors to match model graph expected dimensionality of the loaded weights
    partial_state_dict = load_partial_weights(AutoModelForVision2Seq, model_repo_name, "model.vision_model.")
    reference_model.load_state_dict(partial_state_dict)

    tt_ccl = TT_CCL(mesh_device)
    tt_model = TtLlamaVisionEncoder(
        mesh_device,
        tt_ccl,
        state_dict,
        first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=model_args,
        return_intermediate=return_intermediate,
    )

    # Create rand inputs of the right shape
    batch, num_media, num_chunks, n_channel, patch_size = (1, 1, 4, 3, model_args.vision_chunk_size)
    images = torch.randn(batch, num_media, num_chunks, n_channel, patch_size, patch_size)
    ars = torch.tensor([2, 2]).reshape(batch, num_media, 2)
    aspect_ratio_ids = torch.from_numpy(
        convert_aspect_ratios_to_ids(ars, max_image_tiles=config.vision_config.max_num_tiles)
    )
    aspect_ratio_mask = torch.from_numpy(
        build_aspect_ratio_mask(ars, max_image_tiles=config.vision_config.max_num_tiles)
    )

    with torch.no_grad():
        reference_output = reference_model(images, aspect_ratio_ids, aspect_ratio_mask)[
            0
        ]  # 0-index is the last hidden state

        tt_out = tt_model(images, ars)
        tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        tt_output_torch = tt_output_torch[0, :, :, :].view(reference_output.shape)

        # reference_output is [x] + [shuffled_int_x]
        # tt_output is [x] + [int_x]
        # To compare, we will shuffle tt_output.
        tt_output_shuffled = torch.zeros_like(tt_output_torch)
        tt_output_shuffled[..., : model_args.vision_dim] = tt_output_torch[..., : model_args.vision_dim]
        tt_int_x = tt_output_torch[..., model_args.vision_dim :]
        tt_int_x = (
            tt_int_x.reshape(reference_output.shape[:-1] + (5, model_args.vision_dim))
            .transpose(-1, -2)
            .reshape(reference_output.shape[:-1] + (model_args.vision_dim * 5,))
        )
        tt_output_shuffled[..., model_args.vision_dim :] = tt_int_x

        passing, pcc_message = comp_pcc(reference_output, tt_output_shuffled, pcc_required)

        logger.info(comp_allclose(reference_output, tt_output_shuffled))
        logger.info(f"PCC: {pcc_message}")

    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"
