"""Gemma-3-4b-it Test for Vision Layernorm"""


# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.multimodal.llama_layernorm import TtLayerNorm  # Updated import for LayerNorm
from models.experimental.gemma3_4b.tests.references import reference_vision_layernorm
from models.common.utility_functions import comp_allclose, comp_pcc, nearest_32, skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("layer_name", [("layer_norm1"), ("layer_norm2")])
def test_layernorm_inference(mesh_device, reset_seeds, layer_name):
    dtype = ttnn.bfloat16

    model_args = ModelArgs(mesh_device)
    width = model_args.vision_dim
    num_chunks = 4
    seq_len = nearest_32(model_args.image_size) * num_chunks

    # Load full state dict
    state_dict = model_args.load_state_dict()

    # Prefix for vision MLP weights — consistent with HF checkpoint
    if layer_name == "layer_norm1":
        first_layer_prefix = "visual.encoder.layers.0.ln_1."
    else:
        first_layer_prefix = "visual.encoder.layers.0.ln_2."

    model_args.WEIGHTS_DTYPE = dtype
    # Reference HF MLP (from Gemma3 vision tower)
    reference_model = reference_vision_layernorm(model_args, layer_name)
    # reference_model.load_state_dict(partial_state_dict)
    reference_model.eval()

    # Initialize the custom LayerNorm model
    tt_model = TtLayerNorm(
        device=mesh_device,
        dim=width,
        state_dict=state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        weight_dtype=dtype,
        eps=model_args.norm_eps,
    )

    # Generate random input
    torch_input = torch.rand(1, seq_len, width)  # Adjusted dimensions for LayerNorm

    # Reference output using PyTorch's LayerNorm
    reference_output = reference_model(torch_input)

    # Convert input to ttnn tensor
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Compilation pass for LayerNorm")
    tt_output = tt_model(tt_input)

    tt_output_torch = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)
    )  # Adjusted dim for LayerNorm
    tt_outputs = torch.chunk(tt_output_torch, model_args.num_devices, dim=-1)

    # Compare outputs
    pcc_required = 0.99
    for idx, tt_output_torch in enumerate(tt_outputs):
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

        non_zero_indices = tt_output_torch.ne(0).nonzero(as_tuple=True)
        tt_output_torch = tt_output_torch[non_zero_indices]
        reference_output = reference_output[non_zero_indices]

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
        assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"
