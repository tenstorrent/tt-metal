# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import ttnn

from ..reference.transformer import SD3Transformer2DModel
from ..tt.transformer import TtSD3Transformer2DModel, TtSD3Transformer2DModelParameters
from ..tt.utils import from_torch, assert_quality
from models.utility_functions import (
    nearest_32,
)


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    ("model_name", "batch_size", "prompt_sequence_length", "height", "width"),
    [
        ("large", 1, 333, 1024, 1024),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192, "trace_region_size": 15157248}], indirect=True)
def test_transformer(
    *,
    mesh_device: ttnn.MeshDevice,
    model_name,
    batch_size: int,
    prompt_sequence_length: int,
    height: int,
    width: int,
) -> None:
    mesh_device.enable_async(True)
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16

    torch_model = SD3Transformer2DModel.from_pretrained(
        f"stabilityai/stable-diffusion-3.5-{model_name}", subfolder="transformer", torch_dtype=torch_dtype
    )
    torch_model.eval()
    if model_name == "medium":
        embedding_dim = 1536
    else:
        embedding_dim = 2432

    parameters = TtSD3Transformer2DModelParameters.from_torch(
        torch_model.state_dict(),
        num_heads=torch_model.config.num_attention_heads,
        embedding_dim=embedding_dim,
        device=mesh_device,
        dtype=ttnn_dtype,
    )

    tt_model = TtSD3Transformer2DModel(parameters, num_heads=torch_model.config.num_attention_heads, device=mesh_device)

    torch.manual_seed(0)
    spatial = torch.randn((batch_size, 16, height // 8, width // 8))
    prompt = torch.randn((batch_size, prompt_sequence_length, 4096))
    pooled_projection = torch.randn((batch_size, 2048))
    timestep = torch.randint(1000, (batch_size,), dtype=torch_dtype)

    tt_spatial = ttnn.from_torch(
        spatial.permute([0, 2, 3, 1]),  # BCYX -> BYXC
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn_dtype,
    )
    TILE_SIZE = 32
    prompt_extra = prompt_sequence_length % TILE_SIZE
    if prompt_extra > 0:
        prompt_padding = TILE_SIZE - prompt_extra
    else:
        prompt_padding = 0
    prompt_padded = torch.nn.functional.pad(prompt, pad=(0, 0, 0, prompt_padding), mode="constant", value=0.0)
    tt_prompt = ttnn.from_torch(
        prompt_padded,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn_dtype,
    )
    tt_pooled_projection = ttnn.from_torch(
        pooled_projection,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    tt_timestep = ttnn.from_torch(
        timestep.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
    )

    with torch.no_grad():
        torch_output = torch_model(
            spatial=spatial, prompt_embed=prompt, pooled_projections=pooled_projection, timestep=timestep
        )

    # tt_spatial = allocate_tensor_on_device_like(tt_spatial_host, device=device)
    # tt_prompt = allocate_tensor_on_device_like(tt_prompt_host, device=device)
    # tt_pooled_projection = allocate_tensor_on_device_like(tt_pooled_projection_host, device=device)
    # tt_timestep = allocate_tensor_on_device_like(tt_timestep_host, device=device)

    # trace = tt_model.cache_and_trace(
    #     spatial=tt_spatial, prompt=tt_prompt, pooled_projection=tt_pooled_projection, timestep=tt_timestep
    # )
    # tt_output = trace(
    #     spatial=tt_spatial_host,
    #     prompt=tt_prompt_host,
    #     pooled_projection=tt_pooled_projection_host,
    #     timestep=tt_timestep_host,
    # )
    tt_output = tt_model(
        spatial=tt_spatial, prompt=tt_prompt, pooled_projection=tt_pooled_projection, timestep=tt_timestep
    )

    print(f"tt_output shape {tt_output.shape} torch_output {torch_output.shape}")
    tt_output = ttnn.squeeze(tt_output, 0)
    assert_quality(torch_output, tt_output, mse=0.1, pcc=0.999_500)
