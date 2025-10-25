# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
import math

from ..reference.transformer import SD3Transformer2DModel
from ..tt.fun_transformer import sd_transformer, TtSD3Transformer2DModelParameters
from ..tt.utils import assert_quality
from ..tt.parallel_config import StableDiffusionParallelManager

TILE_SIZE = 32


@pytest.mark.parametrize(
    (
        "model_name",
        "batch_size",
        "prompt_sequence_length",
        "spatial_sequence_length",
        "height",
        "width",
    ),
    [
        ("large", 1, 352, 4096, 1024, 1024),
    ],
)
@pytest.mark.parametrize(
    (
        "mesh_device",
        "cfg",
        "sp",
        "tp",
        "topology",
        "num_links",
    ),
    [
        [(2, 4), (2, 0), (1, 0), (4, 1), ttnn.Topology.Linear, 1],
        [(2, 4), (2, 1), (2, 0), (2, 1), ttnn.Topology.Linear, 1],
        [(8, 4), (2, 0), (4, 0), (4, 1), ttnn.Topology.Linear, 3],
        [(8, 4), (2, 1), (8, 0), (2, 1), ttnn.Topology.Linear, 3],
        [(8, 4), (2, 1), (2, 1), (8, 0), ttnn.Topology.Linear, 3],
    ],
    ids=[
        "t3k_cfg2_sp1_tp4",
        "t3k_cfg2_sp2_tp2",
        "tg_cfg2_sp4_tp4",
        "tg_cfg2_sp8_tp2",
        "tg_cfg2_sp2_tp8",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 8192, "trace_region_size": 15157248}],
    indirect=True,
)
def test_transformer(
    *,
    mesh_device: ttnn.MeshDevice,
    model_name,
    batch_size: int,
    prompt_sequence_length: int,
    spatial_sequence_length: int,
    height: int,
    width: int,
    cfg: int,
    sp: int,
    tp: int,
    topology: ttnn.Topology,
    num_links: int,
) -> None:
    cfg_factor, cfg_axis = cfg
    sp_factor, sp_axis = sp
    tp_factor, tp_axis = tp
    parallel_manager = StableDiffusionParallelManager(
        mesh_device,
        cfg_factor,
        sp_factor,
        tp_factor,
        sp_factor,
        tp_factor,
        topology,
        cfg_axis=cfg_axis,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
    )
    submesh = parallel_manager.submesh_devices[0]
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16

    torch_model = SD3Transformer2DModel.from_pretrained(
        f"stabilityai/stable-diffusion-3.5-{model_name}",
        subfolder="transformer",
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    embedding_dim = 1536 if model_name == "medium" else 2432

    torch_model.eval()

    ## heads padding
    assert not embedding_dim % torch_model.config.num_attention_heads, "Embedding_dim % num_heads != 0"
    pad_embedding_dim = ((torch_model.config.num_attention_heads) % tp_factor) != 0
    hidden_dim_padding = 0
    if pad_embedding_dim:
        head_size = embedding_dim // torch_model.config.num_attention_heads
        num_heads = math.ceil(torch_model.config.num_attention_heads / tp_factor) * tp_factor
        hidden_dim_padding = (num_heads * head_size) - embedding_dim
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
        device=submesh,
        dtype=ttnn_dtype,
        guidance_cond=guidance_cond,
        parallel_config=parallel_manager.dit_parallel_config,
        height=height // 8,
        width=width // 8,
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

    seq_parallel_shard_dim = 1  # 1 is height
    spatial_dims = [None, None]
    spatial_dims[parallel_manager.dit_parallel_config.sequence_parallel.mesh_axis] = seq_parallel_shard_dim
    tt_spatial = ttnn.from_torch(
        spatial.permute([0, 2, 3, 1]),  # BCYX -> BYXC
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            submesh,
            parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape,
            dims=spatial_dims,
        ),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn_dtype,
    )
    tt_prompt = ttnn.from_torch(
        prompt,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            submesh, parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape, dims=[None, None]
        ),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn_dtype,
    )
    tt_pooled_projection = ttnn.from_torch(
        pooled_projection,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            submesh, parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape, dims=[None, None]
        ),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    tt_timestep = ttnn.from_torch(
        timestep.unsqueeze(1),
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            submesh, parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape, dims=[None, None]
        ),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
    )

    tt_output = sd_transformer(
        spatial=tt_spatial,
        prompt=tt_prompt,
        pooled_projection=tt_pooled_projection,
        timestep=tt_timestep,
        parameters=parameters,
        parallel_manager=parallel_manager,
        num_heads=num_heads,
        N=spatial_sequence_length,
        L=prompt_sequence_length,
        cfg_index=0,
    )

    assert_quality(torch_output, tt_output, pcc=0.997, mse=0.06, shard_dim=0, num_devices=submesh.get_num_devices())
