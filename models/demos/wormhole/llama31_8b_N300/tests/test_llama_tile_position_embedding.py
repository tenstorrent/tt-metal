# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

##### Python imports #####
import math
import pytest
from loguru import logger
import os
import itertools

##### PyTorch imports #####
import torch
import torch.nn.functional as F
import torch.nn as nn

##### TTNN imports #####
import ttnn
from ttnn import experimental as ttl
from ttnn import ConcatMeshToTensor, ReplicateTensorToMesh
from models.utility_functions import skip_for_grayskull
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import (
    nearest_32,
)
from models.demos.wormhole.llama31_8b_N300.tt.llama_tile_position_embedding import (
    TtLlamaTilePositionEmbedding,
)
from models.demos.wormhole.llama31_8b_N300.tt.model_config import TtModelArgs

import importlib

llama_reference_mod = importlib.import_module(
    "models.demos.t3000.llama2_70b.reference.llama-models.models.llama3.reference_impl.multimodal.model"
)


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (2, 4), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "gated",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "input_shape, dim, max_num_tiles",
    [
        ((1, 32, 4, 1032), 1280, 4),
        ((1, 8, 4, 1032), 1280, 4),
        ((1, 4, 4, 1032), 1280, 4),
        ((1, 1, 4, 1032), 1280, 4),
        ((1, 1, 4, 1024), 1280, 4),
        # ((1, 32, 16, 1032), 1280, 16), # Large test, takes some time
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
def test_llama_conv2d_inference(
    mesh_device,
    use_program_cache,
    reset_seeds,
    # Input params
    input_shape,
    layout,
    # Tile Position Embedding params
    dim,
    gated,
    max_num_tiles,
):
    dtype = ttnn.bfloat16
    pcc = 0.9999

    model_args = TtModelArgs(mesh_device)
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = "vision_model.vision_encoder.pre_tile_pos_embed."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }
    num_devices = model_args.num_devices

    bsz, num_concurrent_media, num_chunks, ntok = input_shape

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
    print("Checking outputs")
    out = ttnn.from_device(tt_output)
    tt_output_torch = ttnn.to_torch(out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))

    # Only select output from one device
    tt_output_torch = tt_output_torch[..., :dim]

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info(f"Llama_TilePositionEmbedding Passed!")
    else:
        logger.warning(f"Llama_TilePositionEmbedding Failed!")
        assert passing, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
