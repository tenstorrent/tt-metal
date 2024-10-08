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


##### Torch op #####
class TilePositionEmbedding(nn.Module):
    def __init__(
        self,
        num_tiles: int,
        width: int,
        gated: bool = False,
    ):
        super().__init__()
        self.num_tiles = num_tiles
        self.width = width
        self.embedding = nn.Parameter(torch.randn(num_tiles, num_tiles, 1, width) / math.sqrt(width))
        self.gated = gated
        if gated:
            self.gate = nn.Parameter(torch.randn(1))

    """NOTE: The _dynamic_resize function is only ever used in the load hooks function call that loads the weights.
    Currently, this function is NOT tested by the test suite
    """

    @staticmethod
    def _dynamic_resize(embed: torch.Tensor, num_tiles: int):
        nt_old, nt_old, _, w = embed.shape
        embed = embed.permute(2, 3, 0, 1)

        embed_new = F.interpolate(
            embed,
            size=(num_tiles, num_tiles),
            mode="bilinear",
            align_corners=True,
        )
        # reshape the weights to the correct shape
        embed_new = embed_new.permute(2, 3, 0, 1)
        return embed_new

    def forward(self, x: torch.Tensor, ar: torch.Tensor, num_tiles: int = None):
        embed = self.embedding
        if num_tiles is None:
            num_tiles = self.num_tiles
        elif num_tiles > self.num_tiles:
            embed = TilePositionEmbedding._dynamic_resize(self.embedding, num_tiles)
        out_pos_embed = torch.zeros(x.shape[0], num_tiles, 1, self.width, device=x.device, dtype=x.dtype)
        for idx, arx in enumerate(ar):
            h, w = arx
            out_pos_embed[idx, : w * h] = embed[:h, :w].reshape(w * h, 1, self.width)
        if self.gated:
            out_pos_embed = out_pos_embed * self.gate.tanh()

        x = x + out_pos_embed
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
    "gated",
    [
        True,
        False,
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
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,
    ],
)
def test_llama_conv2d_inference(
    mesh_device,
    use_program_cache,
    reset_seeds,
    # Input params
    input_shape,
    input_dtype,
    layout,
    # Tile Position Embedding params
    dim,
    gated,
    max_num_tiles,
):
    pcc = 0.9999

    devices = mesh_device.get_devices()
    num_devices = len(devices)

    bsz, num_concurrent_media, num_chunks, ntok = input_shape

    ##### Check parms #####
    assert num_chunks == max_num_tiles, "num_chunks must be the same value as max_num_tiles!"

    ##### Prepare inputs #####
    input_tensor = torch.randn(bsz * num_concurrent_media, num_chunks, ntok, dim)
    logger.info(f"Input tensor shape: {input_tensor.shape}")

    tt_input_tensor = ttnn.as_tensor(
        input_tensor,
        dtype=input_dtype,
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
    reference_model = TilePositionEmbedding(
        num_tiles=max_num_tiles,
        width=dim,
        gated=gated,
    )
    reference_output = reference_model(input_tensor, aspect_ratios)

    ##### Perform the TT ops #####
    _embedding_config = {
        "dtype": input_dtype,
        "layout": layout,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        "mesh_mapper": ttnn.ReplicateTensorToMesh,
    }

    tt_model = TtLlamaTilePositionEmbedding(
        mesh_device,
        num_tiles=max_num_tiles,
        width=dim,
        gate=reference_model.gate if gated else None,
        embedding=reference_model.embedding,
        embedding_config=_embedding_config,
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
