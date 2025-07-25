# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest
import torch
import ttnn

from models.experimental.flux.tt.timestep_embedding import (
    CombinedTimestepTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddingsParameters,
)
from models.experimental.flux.tt.utils import assert_quality, from_torch_fast


from models.experimental.flux.reference import FluxTransformer as FluxTransformerReference
from models.experimental.flux.reference.timestep_embedding import (
    CombinedTimestepTextProjEmbeddings as CombinedTimestepTextProjEmbeddingsReference,
)


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE") or "N300", len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("guidance_embeds", [False, True])
def test_timestep_embedding(*, mesh_device: ttnn.MeshDevice, guidance_embeds: bool, model_location_generator) -> None:
    batch_size = 512

    torch.manual_seed(0)

    if guidance_embeds:
        model_name = model_location_generator("black-forest-labs/FLUX.1-dev", model_subdir="Flux1_Dev")
    else:
        model_name = model_location_generator("black-forest-labs/FLUX.1-schnell", model_subdir="Flux1_Schnell")

    flux_model = FluxTransformerReference.from_pretrained(
        model_name,
        subfolder="transformer",
        guidance_embeds=guidance_embeds,
    )

    torch_model: CombinedTimestepTextProjEmbeddingsReference = flux_model.time_text_embed

    parameters = CombinedTimestepTextProjEmbeddingsParameters.from_torch(
        torch_model.state_dict(), device=mesh_device, dtype=ttnn.bfloat8_b, guidance_embeds=guidance_embeds
    )
    tt_model = CombinedTimestepTextProjEmbeddings(parameters)

    timestep = torch.tensor([500], dtype=torch.float32)
    pooled_projection = torch.randn((batch_size, 768))
    guidance_projections = torch.tensor([3.5], dtype=torch.float32) if guidance_embeds else None

    unsharded = ttnn.ReplicateTensorToMesh(mesh_device)
    batch_sharded = ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), (0, None))

    tt_timestep = from_torch_fast(
        timestep.unsqueeze(1),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
        mesh_mapper=unsharded,
    )
    tt_pooled_projection = from_torch_fast(
        pooled_projection,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=batch_sharded,
    )

    tt_guidance_projections = None
    if guidance_embeds:
        tt_guidance_projections = from_torch_fast(
            guidance_projections.unsqueeze(1),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.float32,
            mesh_mapper=unsharded,
        )

    with torch.no_grad():
        torch_output = torch_model(timestep, pooled_projection, guidance_projections)

    tt_output = tt_model.forward(
        timestep=tt_timestep,
        pooled_projection=tt_pooled_projection,
        guidance=tt_guidance_projections,
    )

    # Convert to torch tensor and ensure shape matches
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, tuple(mesh_device.shape), (0, -1)),
    )
    if len(tt_output_torch.shape) > 2:
        tt_output_torch = tt_output_torch.reshape(tt_output_torch.shape[0], -1)
    tt_output_torch = tt_output_torch[..., : torch_output.shape[-1]]

    assert_quality(torch_output, tt_output_torch, pcc=0.997, mse=0.1)
