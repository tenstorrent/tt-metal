# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.flux.reference.transformer import FluxTransformer as FluxTransformerReference
from models.experimental.flux.tt.transformer import FluxTransformer, FluxTransformerParameters
from models.experimental.flux.tt.utils import assert_quality


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
    ("spatial_sequence_length", "prompt_sequence_length", "block_count", "pcc", "mse"),
    [
        # (1024, 512, None, 0.99944, 13.8),
        (4096, 512, None, 0.984, 19),
        # (4096, 512, 1, 0.992, 320),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192, "trace_region_size": 15157248}], indirect=True)
def test_transformer(
    *,
    mesh_device: ttnn.MeshDevice,
    prompt_sequence_length: int,
    spatial_sequence_length: int,
    block_count: int | None,
    pcc: float,
    mse: float,
    model_location_generator,
) -> None:
    mesh_height, _ = mesh_device.shape
    batch_size = 1

    torch.manual_seed(0)

    logger.info("loading model...")

    model_name = model_location_generator("black-forest-labs/FLUX.1-schnell", model_subdir="Flux1_Schnell")

    torch_model = FluxTransformerReference.from_pretrained(model_name, subfolder="transformer")
    torch_model.eval()
    torch_model.keep_blocks_only(block_count, block_count)

    spatial = torch.randn([batch_size, spatial_sequence_length, 64])
    prompt = torch.randn([batch_size, prompt_sequence_length, 4096])
    pooled_projection = torch.randn([batch_size, 768])
    timestep = torch.tensor([500], dtype=torch.float32)
    imagerot1 = torch.randn([spatial_sequence_length + prompt_sequence_length, 128])
    imagerot2 = torch.randn([spatial_sequence_length + prompt_sequence_length, 128])

    logger.info("running PyTorch model...")
    with torch.no_grad():
        torch_output = torch_model(
            spatial=spatial,
            prompt_embed=prompt,
            pooled_projections=pooled_projection,
            timestep=timestep,
            image_rotary_emb=(imagerot1, imagerot2),
        )

    del torch_model

    logger.info("loading model...")
    torch_model_bfloat16 = FluxTransformerReference.from_pretrained(
        model_name, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    torch_model_bfloat16.eval()
    torch_model_bfloat16.keep_blocks_only(block_count, block_count)

    logger.info("creating TT-NN model...")
    parameters = FluxTransformerParameters.from_torch(
        torch_model_bfloat16.state_dict(), device=mesh_device, dtype=ttnn.bfloat8_b
    )
    tt_model = FluxTransformer(parameters, num_attention_heads=torch_model_bfloat16.config.num_attention_heads)

    batch_sharded = ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), (0, None))
    unsharded = ttnn.ReplicateTensorToMesh(mesh_device)

    tt_timestep = ttnn.allocate_tensor_on_device([1, 1], ttnn.float32, ttnn.TILE_LAYOUT, mesh_device)

    tt_imagerot1 = ttnn.allocate_tensor_on_device(imagerot1.shape, ttnn.float32, ttnn.TILE_LAYOUT, mesh_device)
    tt_imagerot2 = ttnn.allocate_tensor_on_device(imagerot2.shape, ttnn.float32, ttnn.TILE_LAYOUT, mesh_device)

    tt_pooled_prompt_embeds = ttnn.from_torch(
        pooled_projection,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), (0, None)),
    )

    tt_spatial = ttnn.from_torch(
        spatial,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), (0, -2)),
    )
    tt_prompt = ttnn.from_torch(
        prompt,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), (0, -1)),
    )

    tt_timestep = ttnn.from_torch(
        timestep.unsqueeze(1), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, mesh_mapper=unsharded
    )
    tt_imagerot1 = ttnn.from_torch(
        imagerot1, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, mesh_mapper=unsharded
    )
    tt_imagerot2 = ttnn.from_torch(
        imagerot2, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, mesh_mapper=unsharded
    )

    model_args = dict(  # noqa: C408
        spatial=tt_spatial,
        prompt=tt_prompt,
        pooled_projection=tt_pooled_prompt_embeds,
        timestep=tt_timestep,
        image_rotary_emb=(tt_imagerot1, tt_imagerot2),
    )

    logger.info("forward pass...")
    tt_output = tt_model.forward(**model_args)

    composer = ttnn.ConcatMesh2dToTensor(mesh_device, tuple(mesh_device.shape), (0, -2))
    assert_quality(torch_output, tt_output, pcc=pcc, mse=mse, mesh_composer=composer)
