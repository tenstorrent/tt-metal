# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING
import os

import pytest
import torch
import ttnn
from loguru import logger

from ..tt import utils
from ..tt.transformer_block import TransformerBlock, TransformerBlockParameters
from ..tt.utils import assert_quality

if TYPE_CHECKING:
    from ..reference.transformer_block import TransformerBlock as TransformerBlockReference


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE") or "N300", len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    ("block_index", "spatial_sequence_length", "prompt_sequence_length"),
    [
        (0, 4096, 512),
    ],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 716800}], indirect=True)
@pytest.mark.parametrize("use_tracing", [False])  # Tracing currently causes a mesh device to hang.
def test_transformer_block(
    *,
    mesh_device: ttnn.MeshDevice,
    use_tracing: bool,
    block_index: int,
    spatial_sequence_length: int,
    prompt_sequence_length: int,
    parent_torch_model: FluxTransformer,
) -> None:
    batch_size, _ = mesh_device.shape

    torch.manual_seed(0)

    torch_model: TransformerBlockReference = parent_torch_model.transformer_blocks[block_index].to(torch.float32)

    logger.debug("creating TT-NN model...")
    parameters = TransformerBlockParameters.from_torch(
        torch_model.state_dict(), device=mesh_device, dtype=ttnn.bfloat8_b
    )
    tt_model = TransformerBlock(parameters, num_heads=torch_model.num_heads)

    embedding_dim = 3072

    spatial = torch.randn((batch_size, spatial_sequence_length, embedding_dim))
    prompt = torch.randn((batch_size, prompt_sequence_length, embedding_dim))
    time = torch.randn((batch_size, embedding_dim))
    imagerot1 = torch.randn([spatial_sequence_length + prompt_sequence_length, 128], dtype=torch.float32)
    imagerot2 = torch.randn([spatial_sequence_length + prompt_sequence_length, 128], dtype=torch.float32)

    sharded = ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), (0, -1))
    batch_sharded = ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), (0, None))
    unsharded = ttnn.ReplicateTensorToMesh(mesh_device)

    tt_spatial = ttnn.from_torch(
        spatial, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=sharded
    )
    tt_prompt = ttnn.from_torch(
        prompt, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, mesh_mapper=sharded
    )
    tt_time = ttnn.from_torch(
        time.unsqueeze(1), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=batch_sharded
    )
    tt_imagerot1 = ttnn.from_torch(
        imagerot1, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, mesh_mapper=unsharded
    )
    tt_imagerot2 = ttnn.from_torch(
        imagerot2, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, mesh_mapper=unsharded
    )

    with torch.no_grad():
        spatial_output, prompt_output = torch_model(
            spatial=spatial, prompt=prompt, time_embed=time, image_rotary_emb=(imagerot1, imagerot2)
        )

    model_args = dict(  # noqa: C408
        spatial=tt_spatial,
        prompt=tt_prompt,
        time_embed=tt_time,
        image_rotary_emb=(tt_imagerot1, tt_imagerot2),
    )

    # compile
    logger.debug("compiling...")
    tt_model.forward(**model_args)

    # execute
    logger.debug("executing...")

    utils.signpost("start")
    tt_spatial_output, tt_prompt_output = tt_model.forward(**model_args)
    utils.signpost("end")

    assert (prompt_output is None) == (tt_prompt_output is None)

    composer = ttnn.ConcatMesh2dToTensor(mesh_device, tuple(mesh_device.shape), (0, -1))

    if tt_prompt_output is not None:
        assert_quality(prompt_output, tt_prompt_output, pcc=0.99929, mse=1500, mesh_composer=composer)
    assert_quality(spatial_output, tt_spatial_output, pcc=0.9988, mse=83, mesh_composer=composer)
