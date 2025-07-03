# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest
import torch
import ttnn

from ..tt.timestep_embedding import CombinedTimestepTextProjEmbeddings, CombinedTimestepTextProjEmbeddingsParameters
from ..tt.utils import assert_quality, from_torch_fast


from ..reference import FluxTransformer as FluxTransformerReference
from ..reference.timestep_embedding import (
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
def test_timestep_embedding(*, mesh_device: ttnn.MeshDevice) -> None:
    batch_size = 512

    torch.manual_seed(0)

    flux_model = FluxTransformerReference()

    torch_model: CombinedTimestepTextProjEmbeddingsReference = flux_model.time_text_embed.to(torch.float32)

    parameters = CombinedTimestepTextProjEmbeddingsParameters.from_torch(
        torch_model.state_dict(), device=mesh_device, dtype=ttnn.bfloat8_b
    )
    tt_model = CombinedTimestepTextProjEmbeddings(parameters)

    timestep = torch.tensor([500], dtype=torch.float32)
    pooled_projection = torch.randn((batch_size, 768))

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

    with torch.no_grad():
        torch_output = torch_model(timestep, pooled_projection)

    tt_output = tt_model.forward(timestep=tt_timestep, pooled_projection=tt_pooled_projection)
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, tuple(mesh_device.shape), (0, -1)),
    )[..., : torch_output.shape[-1]]

    assert_quality(torch_output, tt_output_torch, pcc=0.997, mse=0.1)
