# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
from typing import Optional

import pytest
import torch
from loguru import logger
from torch import nn
from transformers import AutoConfig, AutoModelForVision2Seq
from transformers.models.mllama.configuration_mllama import MllamaConfig
from transformers.models.mllama.image_processing_mllama import build_aspect_ratio_mask, convert_aspect_ratios_to_ids
from transformers.models.mllama.modeling_mllama import MllamaPreTrainedModel, MllamaVisionModel

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tests.multimodal.utils import load_partial_weights
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.multimodal.llama_cross_attention_transformer_vision import (
    TtLlamaCrossAttentionTransformerVision,
)


# Meta's lib crossattention vision transformer branch found in :https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3/multimodal/model.py#L1027
# does not exist indepedently as a subclass in HF, instead it is implemented directly in the whole mLlama model graph.
# Thus the following custom class is adapted from mLlama model of HF https://github.com/huggingface/transformers/blob/d12672fc50c1393ef3bb0964aaebd4290d93e364/src/transformers/models/mllama/modeling_mllama.py#L1445
# to create the same computational graph as in meta lib for this comparison with tt_model containing the vision model and the mutlimodal projector.
class CrossAttentionTransformerVisionModel(MllamaPreTrainedModel):
    def __init__(self, config: MllamaConfig):
        super().__init__(config)
        self.vocab_size = config.text_config.vocab_size
        self.hidden_size = config.text_config.hidden_size
        self.max_num_tiles = config.vision_config.max_num_tiles
        self.vision_output_dim = config.vision_config.vision_output_dim

        self.vision_model = MllamaVisionModel(config.vision_config)
        self.multi_modal_projector = nn.Linear(
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            bias=True,
        )
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
        aspect_ratio_mask: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pixel_values is not None and cross_attention_states is not None:
            raise ValueError("`pixel_values` and `cross_attention_states` cannot be provided simultaneously")

        if pixel_values is not None:
            if aspect_ratio_ids is None:
                raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
            # get vision tokens from vision model
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
            )
            cross_attention_states = vision_outputs[0]
            cross_attention_states = self.multi_modal_projector(cross_attention_states).reshape(
                -1, cross_attention_states.shape[-2], self.hidden_size
            )

        return cross_attention_states


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
def test_vision_transformer_inference(mesh_device, reset_seeds):
    dtype = ttnn.bfloat16
    pcc_required = 0.79

    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = "vision_model."

    return_intermediate = "3,7,15,23,30"
    return_intermediate = [int(l) for l in return_intermediate.split(",")]

    model_repo_name = os.getenv("HF_MODEL")
    # config contains parameters for the whole multimodal network the subset of vision branch is chosen instead
    config = AutoConfig.from_pretrained(model_repo_name)
    config.vision_config._attn_implementation = "sdpa"
    reference_model = CrossAttentionTransformerVisionModel(config)

    # Partial loading of HF safetensors to match model graph expected dimensionality of the loaded weights.
    # Because the custom class CrossAttentionTransformerVisionModel wraps the vision_model and multi_modal_projector,
    # the following prefixes need to be added so the weights can be loaded correctly on the computational graph.
    add_prefix = lambda d, prefix: {f"{prefix}{k}": v for k, v in d.items()}
    partial_state_dict = add_prefix(
        load_partial_weights(AutoModelForVision2Seq, model_repo_name, "model.vision_model."),
        "vision_model.",
    )

    multimodal_proj_weights = add_prefix(
        load_partial_weights(AutoModelForVision2Seq, model_repo_name, "model.multi_modal_projector."),
        "multi_modal_projector.",
    )

    partial_state_dict.update(multimodal_proj_weights)
    reference_model.load_state_dict(partial_state_dict)

    tt_ccl = TT_CCL(mesh_device)
    tt_model = TtLlamaCrossAttentionTransformerVision(
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
    chunk_seq_len = model_args.vision_chunk_ntok  # tokens per chunk, including class token
    images = torch.randn(batch, num_media, num_chunks, n_channel, patch_size, patch_size)
    ars = torch.tensor([2, 2]).reshape(batch, num_media, 2)
    aspect_ratio_ids = torch.from_numpy(
        convert_aspect_ratios_to_ids(ars, max_image_tiles=config.vision_config.max_num_tiles)
    )
    aspect_ratio_mask = torch.from_numpy(
        build_aspect_ratio_mask(ars, max_image_tiles=config.vision_config.max_num_tiles)
    )

    with torch.no_grad():
        reference_output = reference_model(images, aspect_ratio_ids, aspect_ratio_mask)
        tt_out = tt_model(images, ars)
        tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        tt_output_torch = tt_output_torch[0, :, :chunk_seq_len, :].view(reference_output.shape)

        logger.info(f"Reference output shape: {reference_output.shape}")
        logger.info(f"TT output shape: {tt_output_torch.shape}")

        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
        assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"
