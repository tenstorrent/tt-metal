# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import ttnn

from ..reference.transformer import SD3Transformer2DModel
from ..tt.fun_transformer import sd_transformer, TtSD3Transformer2DModelParameters
from ..tt.utils import assert_quality
from ..tt.parallel_config import create_dit_parallel_config, ParallelConfig

TILE_SIZE = 32


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
    ("model_name", "batch_size", "prompt_sequence_length", "spatial_sequence_length", "height", "width"),
    [
        ("large", 2, 333, 4096, 1024, 1024),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192, "trace_region_size": 15157248}], indirect=True)
@pytest.mark.usefixtures("use_program_cache")
def test_transformer(
    *,
    mesh_device: ttnn.MeshDevice,
    model_name,
    batch_size: int,
    prompt_sequence_length: int,
    spatial_sequence_length: int,
    height: int,
    width: int,
) -> None:
    mesh_shape = tuple(mesh_device.shape)
    cfg_parallel = ParallelConfig(mesh_shape=mesh_shape, factor=1, mesh_axis=0)
    tensor_parallel = ParallelConfig(mesh_shape=(mesh_shape[0], 1), factor=mesh_shape[1], mesh_axis=1)
    dit_parallel_config = create_dit_parallel_config(
        mesh_shape=mesh_shape, cfg_parallel=cfg_parallel, tensor_parallel=tensor_parallel
    )
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

    num_devices = mesh_device.get_num_devices()
    ## heads padding for T3K TP
    pad_embedding_dim = False
    if os.environ["MESH_DEVICE"] == "T3K" and embedding_dim == 2432:
        pad_embedding_dim = True
        hidden_dim_padding = (
            ((embedding_dim // num_devices // TILE_SIZE) + 1) * TILE_SIZE
        ) * num_devices - embedding_dim
        num_heads = 40
    else:
        num_heads = torch_model.config.num_attention_heads

    guidance_cond = 1
    if batch_size == 2:
        guidance_cond = 2

    parameters = TtSD3Transformer2DModelParameters.from_torch(
        torch_model.state_dict(),
        num_heads=num_heads,
        unpadded_num_heads=torch_model.config.num_attention_heads,
        embedding_dim=embedding_dim,
        hidden_dim_padding=hidden_dim_padding,
        device=mesh_device,
        dtype=ttnn_dtype,
        guidance_cond=guidance_cond,
        parallel_config=dit_parallel_config,
    )

    torch.manual_seed(0)
    spatial = torch.randn((batch_size, 16, height // 8, width // 8))
    prompt = torch.randn((batch_size, prompt_sequence_length, 4096))
    pooled_projection = torch.randn((batch_size, 2048))
    timestep = torch.randint(1000, (batch_size,), dtype=torch_dtype)

    with torch.no_grad():
        torch_output = torch_model(
            spatial=spatial,
            prompt_embed=prompt,
            pooled_projections=pooled_projection,
            timestep=timestep,
        )

    tt_spatial = ttnn.from_torch(
        spatial.permute([0, 2, 3, 1]),  # BCYX -> BYXC
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn_dtype,
    )
    tt_prompt = ttnn.from_torch(
        prompt,
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
    tt_output = sd_transformer(
        spatial=tt_spatial,
        prompt=tt_prompt,
        pooled_projection=tt_pooled_projection,
        timestep=tt_timestep,
        parameters=parameters,
        parallel_config=dit_parallel_config,
        num_heads=num_heads,
        N=spatial_sequence_length,
        L=prompt_sequence_length,
    )

    torch_output = torch.unsqueeze(torch_output, 1)
    assert_quality(torch_output, tt_output, pcc=0.997, mse=0.06, shard_dim=0, num_devices=mesh_device.get_num_devices())
