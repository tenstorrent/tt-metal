# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs

from models.experimental.mistral_24b.tt.vision_pixtral_transformer import TtPixtralTransformer
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "batch, num_chunks",
    ((1, 1),),
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_image_transformer_inference(batch, num_chunks, mesh_device):
    pcc_required = 0.99

    model_args = ModelArgs(mesh_device)
    dtype = ttnn.bfloat16

    state_dict = model_args.load_state_dict()
    n_layers = model_args.vision_n_layers
    first_layer_prefix = "vision_tower.transformer."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    dim = model_args.vision_dim
    heads = model_args.vision_attn_n_heads
    seq_len = model_args.vision_chunk_ntok - 1
    head_dim = dim // heads

    reference_model = model_args.reference_vision_encoder()
    reference_model.load_state_dict(partial_state_dict)
    reference_model.eval()

    all_tests_pass = True

    tt_model = TtPixtralTransformer(
        mesh_device,
        state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=model_args,
        layers=n_layers,
    )

    # Create PT input
    pt_attention_input = torch.randn(batch, seq_len, dim)
    attention_mask = torch.zeros(batch, 1, seq_len, seq_len)

    B, T, D = pt_attention_input.shape
    cos = torch.ones((1, T, head_dim))
    sin = torch.zeros((1, T, head_dim))

    positional_embedding = (cos, sin)

    attention_input = model_args.prepare_residual_tensor_prefill(
        pt_attention_input,
        force_replicated=True,
    )
    tt_mask = ttnn.from_torch(
        attention_mask,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    with torch.no_grad():
        tt_out = tt_model(attention_input, mask=tt_mask)
        reference_output = reference_model(
            pt_attention_input, attention_mask=attention_mask, position_embeddings=positional_embedding
        )[0]
        tt_output_torch = ttnn.to_torch(tt_out)
        tt_output_torch = tt_output_torch.squeeze(0)
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)
        if not passing:
            logger.warning(f"PCC value -- {pcc_message} -- is lower than {pcc_required} for the output.")
        else:
            logger.info(f"PCC: {pcc_message}")
        logger.info(comp_allclose(reference_output, tt_output_torch))
        all_tests_pass = all_tests_pass and passing

        assert all_tests_pass, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"
