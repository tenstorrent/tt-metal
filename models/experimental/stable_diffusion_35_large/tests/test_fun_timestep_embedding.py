# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import pytest
import torch
import ttnn
from ..reference import SD3Transformer2DModel
from ..tt.fun_timestep_embedding import TtCombinedTimestepTextProjEmbeddingsParameters, sd_combined_timestep_embed
from ..tt.utils import assert_quality, to_torch
from ..tt.parallel_config import StableDiffusionParallelManager

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
    [(1, 8)],
    indirect=True,
)
def test_timestep_embedding(
    *,
    mesh_device: ttnn.MeshDevice,
    model_name,
    batch_size: int,
) -> None:
    parallel_manager = StableDiffusionParallelManager(mesh_device, 1, 1, 1, 1, 1, ttnn.Topology.Linear)
    dtype = torch.bfloat16

    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        f"stabilityai/stable-diffusion-3.5-{model_name}", subfolder="transformer", torch_dtype=dtype
    )
    torch_model: CombinedTimestepTextProjEmbeddings = parent_torch_model.time_text_embed
    torch_model.eval()

    parameters = TtCombinedTimestepTextProjEmbeddingsParameters.from_torch(
        torch_model.state_dict(),
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        guidance_cond=batch_size,
        hidden_dim_padding=0,
        parallel_config=parallel_manager.dit_parallel_config,
    )

    torch.manual_seed(0)
    timestep = torch.randint(1000, (batch_size,), dtype=torch.float32)
    pooled_projection = torch.randn((batch_size, 2048), dtype=dtype)

    tt_timestep = ttnn.from_torch(timestep.unsqueeze(1), device=mesh_device, layout=ttnn.TILE_LAYOUT)
    tt_pooled_projection = ttnn.from_torch(pooled_projection, device=mesh_device, layout=ttnn.TILE_LAYOUT)

    torch_output = torch_model(timestep, pooled_projection)

    tt_output = sd_combined_timestep_embed(
        timestep=tt_timestep, pooled_projection=tt_pooled_projection, parameters=parameters
    )

    tt_output_torch = to_torch(tt_output, mesh_device=mesh_device, dtype=dtype, shard_dim=0).squeeze()[
        :batch_size, : torch_model.timestep_embedder.linear_1.out_features
    ]

    assert_quality(torch_output, tt_output_torch, pcc=0.999_900, num_devices=mesh_device.get_num_devices(), shard_dim=0)
