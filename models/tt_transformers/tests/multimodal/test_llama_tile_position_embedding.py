# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import llama_models.llama3.reference_impl.multimodal.model as llama_reference_mod
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.multimodal.llama_tile_position_embedding import TtLlamaTilePositionEmbedding
from models.utility_functions import comp_allclose, comp_pcc, nearest_32, skip_for_grayskull
from ttnn import ConcatMeshToTensor, ReplicateTensorToMesh


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
@pytest.mark.parametrize(
    "bsz, num_concurrent_media, num_chunks",
    [
        (1, 1, 4),
        (1, 4, 4),
    ],
)
@pytest.mark.parametrize("pre_embed", [False, True])
def test_conv2d_inference(
    mesh_device,
    reset_seeds,
    # Input params
    bsz,
    num_concurrent_media,
    num_chunks,
    pre_embed,
    ensure_gc,
):
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    gated = True
    pcc_required = 0.9999

    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = "vision_model.vision_encoder." + (
        "pre_tile_pos_embed." if pre_embed else "post_tile_pos_embed."
    )
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    ntok = nearest_32(model_args.vision_chunk_ntok - (0 if pre_embed else 1))
    dim = model_args.vision_dim
    max_num_tiles = model_args.vision_max_num_tiles

    ##### Check parms #####
    assert num_chunks == max_num_tiles, "num_chunks must be the same value as max_num_tiles!"

    ##### Prepare inputs #####
    input_tensor = torch.randn(bsz * num_concurrent_media, num_chunks, ntok, dim)
    logger.info(f"Input tensor shape: {input_tensor.shape}")

    tt_input_tensor = ttnn.as_tensor(
        input_tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )
    logger.info(f"TT Input tensor shape: {tt_input_tensor.shape}")

    # Generate all possible aspect ratios (H * W must be less than or equal to max_num_tiles)
    aspect_ratios = list(itertools.product(range(1, max_num_tiles + 1), repeat=2))
    aspect_ratios = [x for x in aspect_ratios if x[0] * x[1] <= max_num_tiles]

    # Repeat the aspect ratios to match the batch size
    if len(aspect_ratios) < bsz * num_concurrent_media:
        aspect_ratios = aspect_ratios * (bsz * num_concurrent_media // len(aspect_ratios) + 1)

    aspect_ratios = torch.tensor(aspect_ratios[: bsz * num_concurrent_media], dtype=torch.int64)
    logger.info(f"Aspects ratios shape: {aspect_ratios.shape}")

    tt_aspect_ratios = aspect_ratios.tolist()

    ##### Perform the torch ops #####
    reference_model = llama_reference_mod.TilePositionEmbedding(
        num_tiles=max_num_tiles,
        width=dim,
        gated=gated,
    )
    reference_model.load_state_dict(partial_state_dict)
    reference_output = reference_model(input_tensor, aspect_ratios)

    ##### Perform the TT ops #####
    tt_model = TtLlamaTilePositionEmbedding(
        mesh_device,
        state_dict=state_dict,
        state_dict_prefix=first_layer_prefix,
        dtype=dtype,
        num_tiles=max_num_tiles,
        width=dim,
        gated=gated,
    )
    tt_output = tt_model(tt_input_tensor, tt_aspect_ratios)

    ##### Check the outputs #####
    logger.info("Checking outputs")
    out = ttnn.from_device(tt_output)
    tt_output_torch = ttnn.to_torch(out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))

    # Only select output from one device
    tt_output_torch = tt_output_torch[..., :dim]

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"
