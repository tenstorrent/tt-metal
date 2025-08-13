# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.experimental.mistral_24b.tt.vision_pixtral_image_block import TtPixtralImageTransformerBlock
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
def test_pixtral_image_block(batch, num_chunks, mesh_device, reset_seeds):
    dtype = ttnn.bfloat16
    pcc_required = 0.99

    model_args = ModelArgs(mesh_device)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    first_layer_prefix = "vision_tower.transformer.layers.0."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    print("partial_state_dict keys:", partial_state_dict.keys())

    dim = model_args.vision_dim
    heads = model_args.vision_attn_n_heads
    seq_len = model_args.vision_chunk_ntok - 1
    head_dim = dim // heads

    reference_model = model_args.reference_pixtral_image_block()
    reference_model.load_state_dict(partial_state_dict)

    tt_model = TtPixtralImageTransformerBlock(
        mesh_device=mesh_device,
        state_dict=state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=model_args,
    )

    pt_attention_input = torch.randn(batch, seq_len, dim).to(torch.bfloat16)
    attention_mask = torch.zeros(batch, 1, seq_len, seq_len).to(torch.bfloat16)

    B, T, D = pt_attention_input.shape
    cos = torch.ones((1, T, head_dim)).to(torch.bfloat16)
    sin = torch.zeros((1, T, head_dim)).to(torch.bfloat16)

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

    tt_out = tt_model(attention_input, mask=tt_mask)
    reference_output = reference_model(
        pt_attention_input, attention_mask=attention_mask, position_embeddings=positional_embedding
    )[0]

    print("tt_out shape:", tt_out.shape)
    print("reference_output shape:", reference_output.shape)

    tt_output_torch = ttnn.to_torch(tt_out).squeeze(0)
    print("tt_output_torch shape:", tt_output_torch.shape)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"
