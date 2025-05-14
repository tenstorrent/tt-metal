# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
import torch
import torch.nn as nn
from loguru import logger

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.multimodal.llama_positional_embedding import TtLlamaPositionalEmbedding
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull
from ttnn import ConcatMeshToTensor, ReplicateTensorToMesh


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
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "bsz, num_concurrent_media, num_chunks",
    [(1, 4, 4)],
)
def test_positional_embedding_inference(
    mesh_device,
    use_program_cache,
    reset_seeds,
    # Input params
    bsz,
    num_concurrent_media,
    num_chunks,
    ensure_gc,
):
    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    pcc_required = 0.9999

    model_args = ModelArgs(mesh_device)
    state_dict = model_args.load_state_dict()
    first_layer_prefix = "vision_model.vision_encoder."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    ntok = model_args.vision_chunk_ntok
    dim = model_args.vision_dim
    image_size = (model_args.vision_chunk_size, model_args.vision_chunk_size)
    patch_size = (model_args.vision_patch_size, model_args.vision_patch_size)

    ##### Check parms #####
    max_num_tiles = model_args.vision_max_num_chunks
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

    tt_input_tensor = ttnn.to_layout(tt_input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    tt_input_tensor = ttnn.reshape(tt_input_tensor, (bsz * num_concurrent_media, num_chunks, ntok, dim))
    tt_input_tensor = ttnn.to_layout(tt_input_tensor, ttnn.TILE_LAYOUT)
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
    reference_model.load_state_dict(partial_state_dict, strict=False)
    reference_output = reference_model(input_tensor, aspect_ratios)

    ##### Perform the TT ops #####
    tt_model = TtLlamaPositionalEmbedding(
        mesh_device,
        state_dict,
        first_layer_prefix,
        None,
        dtype,
        model_args,
    )
    tt_output = tt_model(tt_input_tensor, tt_aspect_ratios)

    ##### Check the outputs #####
    logger.info("Checking outputs")
    out = ttnn.from_device(tt_output)
    tt_output_torch = ttnn.to_torch(out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=-1))

    # Only select output from one device
    tt_output_torch = tt_output_torch[..., :dim]

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"
