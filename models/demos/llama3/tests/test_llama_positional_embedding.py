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
from models.demos.llama3.tt.llama_positional_embedding import (
    TtLlamaPositionalEmbedding,
)
from models.demos.llama3.tt.model_config import TtModelArgs

import importlib


##### Torch op #####
class PositionalEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, max_num_tiles, width):
        super().__init__()

        self.grid_size = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )

        scale = width**-0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))

        self.gated_positional_embedding = nn.Parameter(
            scale
            * torch.randn(
                max_num_tiles,
                max_num_tiles,
                self.grid_size[0] * self.grid_size[1] + 1,
                width,
            )
        )
        self.gated_positional_embedding_gate = nn.Parameter(torch.randn(1))

    def forward(self, x, ar):
        assert x.shape[2] == (self.grid_size[0] * self.grid_size[1] + 1), "Input tensor shape is not correct!"
        # apply regular position embedding
        bsz, num_chunks, num_tokens, dim = x.shape
        x = x.view(bsz * num_chunks, num_tokens, dim)

        x = x + self.positional_embedding * (1 - self.gated_positional_embedding_gate.tanh())
        x = x.view(bsz, num_chunks, num_tokens, dim)

        for idx, arx in enumerate(ar):
            _pos_embed = self.gated_positional_embedding[: arx[0], : arx[1]]
            _pos_embed = _pos_embed.reshape(arx[0] * arx[1], *_pos_embed.shape[2:])
            x[idx, : arx[0] * arx[1]] += _pos_embed * self.gated_positional_embedding_gate.tanh()
        return x


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
    "image_size, patch_size",
    [
        ((448, 448), (14, 14)),
    ],
)
@pytest.mark.parametrize(
    "input_shape, max_num_tiles",
    [
        ((1, 4, 4, 1024 + 1, 1280), 4),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
def test_llama_positional_embedding_inference(
    mesh_device,
    use_program_cache,
    reset_seeds,
    # Input params
    input_shape,
    layout,
    # Positional Embedding params
    image_size,
    patch_size,
    max_num_tiles,
):
    dtype = ttnn.bfloat16
    pcc = 0.9999

    devices = mesh_device.get_devices()
    num_devices = len(devices)

    (
        bsz,
        num_concurrent_media,
        num_chunks,
        ntok,
        dim,
    ) = input_shape

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
    reference_model = PositionalEmbedding(
        image_size=image_size,
        patch_size=patch_size,
        max_num_tiles=max_num_tiles,
        width=dim,
    )
    reference_output = reference_model(input_tensor, aspect_ratios)

    ##### Perform the TT ops #####
    tt_model = TtLlamaPositionalEmbedding(
        mesh_device,
        positional_embedding=reference_model.positional_embedding,
        gated_positional_embedding=reference_model.gated_positional_embedding,
        gated_positional_embedding_gate=reference_model.gated_positional_embedding_gate,
        dtype=dtype,
    )
    tt_output = tt_model(tt_input_tensor, tt_aspect_ratios)

    ##### Check the outputs #####
    logger.info("Checking outputs")
    out = ttnn.from_device(tt_output)
    tt_output_torch = ttnn.to_torch(out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=-1))

    # Only select output from one device
    tt_output_torch = tt_output_torch[..., :dim]

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    if passing:
        logger.info(f"Llama_PositionalEmbedding Passed!")
    else:
        logger.warning(f"Llama_PositionalEmbedding Failed!")
        assert passing, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
