# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import pytest
import torch
import ttnn
import os
from ..reference import SD3Transformer2DModel
from ..tt.timestep_embedding import (
    TtCombinedTimestepTextProjEmbeddings,
    TtCombinedTimestepTextProjEmbeddingsParameters,
)
from ..tt.utils import assert_quality, to_torch

if TYPE_CHECKING:
    from ..reference.timestep_embedding import CombinedTimestepTextProjEmbeddings


@pytest.mark.parametrize(
    ("model_name", "batch_size"),
    [
        ("large", 100),
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("use_program_cache")
def test_timestep_embedding(
    *,
    mesh_device: ttnn.MeshDevice,
    model_name,
    batch_size: int,
) -> None:
    dtype = torch.bfloat16

    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        f"stabilityai/stable-diffusion-3.5-{model_name}", subfolder="transformer", torch_dtype=dtype
    )
    torch_model: CombinedTimestepTextProjEmbeddings = parent_torch_model.time_text_embed
    torch_model.eval()

    parameters = TtCombinedTimestepTextProjEmbeddingsParameters.from_torch(
        torch_model.state_dict(), device=mesh_device, dtype=ttnn.bfloat8_b
    )
    tt_model = TtCombinedTimestepTextProjEmbeddings(batch_size=batch_size, parameters=parameters, device=mesh_device)

    torch.manual_seed(0)
    timestep = torch.randint(1000, (batch_size,), dtype=torch.float32)
    pooled_projection = torch.randn((batch_size, 2048), dtype=dtype)

    tt_timestep = ttnn.from_torch(timestep.unsqueeze(1), device=mesh_device, layout=ttnn.TILE_LAYOUT)
    tt_pooled_projection = ttnn.from_torch(pooled_projection, device=mesh_device, layout=ttnn.TILE_LAYOUT)

    torch_output = torch_model(timestep, pooled_projection)

    tt_output = tt_model(timestep=tt_timestep, pooled_projection=tt_pooled_projection)

    tt_output_torch = to_torch(tt_output, mesh_device=mesh_device, dtype=dtype, shard_dim=0).squeeze()[
        :batch_size, : torch_model.timestep_embedder.linear_1.out_features
    ]

    assert_quality(torch_output, tt_output_torch, pcc=0.999_900, num_devices=mesh_device.get_num_devices(), shard_dim=0)
