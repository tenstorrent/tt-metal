# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import ttnn
import math

from ..reference.transformer import SD3Transformer2DModel
from ..tt.fun_transformer import sd_transformer, TtSD3Transformer2DModelParameters
from ..tt.utils import assert_quality, create_global_semaphores, initialize_sd_parallel_config

TILE_SIZE = 32


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (2, 4), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    (
        "model_name",
        "batch_size",
        "prompt_sequence_length",
        "spatial_sequence_length",
        "height",
        "width",
        "cfg_factor",
        "sp_factor",
        "tp_factor",
        "rp_factor",
        "up_factor",
        "topology",
    ),
    [
        ("large", 1, 352, 4096, 1024, 1024, 1, 2, 4, 2, 4, ttnn.Topology.Linear),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 8192, "trace_region_size": 15157248}],
    indirect=True,
)
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
    cfg_factor: int,
    sp_factor: int,
    tp_factor: int,
    rp_factor: int,
    up_factor: int,
    topology: ttnn.Topology,
) -> None:
    mesh_shape = tuple(mesh_device.shape)
    dit_parallel_config = initialize_sd_parallel_config(
        mesh_shape, cfg_factor, sp_factor, tp_factor, rp_factor, up_factor, topology
    )
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )

    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)

    # create global semaphore handles
    num_devices = mesh_device.get_num_devices()
    ag_ccl_semaphore_handle = create_global_semaphores(mesh_device, num_devices, ccl_sub_device_crs, 0)
    rs_from_ccl_semaphore_handle = create_global_semaphores(mesh_device, num_devices, ccl_sub_device_crs, 0)
    rs_to_ccl_semaphore_handle = create_global_semaphores(mesh_device, num_devices, ccl_sub_device_crs, 0)
    ring_attention_semaphore_handles = [
        create_global_semaphores(mesh_device, num_devices, ccl_sub_device_crs, 0) for _ in range(2)
    ]

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
    pad_embedding_dim = (bool)(torch_model.config.num_attention_heads) % tp_factor
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
        device=mesh_device,
        dtype=ttnn_dtype,
        guidance_cond=guidance_cond,
        parallel_config=dit_parallel_config,
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
    tt_spatial = ttnn.from_torch(
        spatial.permute([0, 2, 3, 1]),  # BCYX -> BYXC
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dit_parallel_config.cfg_parallel.mesh_shape, dims=[seq_parallel_shard_dim, None]
        ),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn_dtype,
    )
    tt_prompt = ttnn.from_torch(
        prompt,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dit_parallel_config.cfg_parallel.mesh_shape, dims=[None, None]),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn_dtype,
    )
    tt_pooled_projection = ttnn.from_torch(
        pooled_projection,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dit_parallel_config.cfg_parallel.mesh_shape, dims=[None, None]),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    tt_timestep = ttnn.from_torch(
        timestep.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dit_parallel_config.cfg_parallel.mesh_shape, dims=[None, None]),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
    )

    # Create persistent buffers
    persistent_buffer_shape = [1, num_heads // up_factor, spatial_sequence_length, head_size]
    persistent_buffers = [
        ttnn.from_torch(
            torch.zeros(persistent_buffer_shape),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn_dtype,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, None]),
        )
        for _ in range(2)
    ]

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
        ag_global_semaphore=ag_ccl_semaphore_handle,
        rs_from_global_semaphore=rs_from_ccl_semaphore_handle,
        rs_to_global_semaphore=rs_to_ccl_semaphore_handle,
        ring_attention_semaphore_handles=ring_attention_semaphore_handles,
        persistent_buffers=persistent_buffers,
        worker_sub_device_id=worker_sub_device_id,
    )

    ttnn.synchronize_device(mesh_device)

    # num_measurement_iterations=1
    # profiler.clear()
    # profiler.start(f"run")
    # for i in range(num_measurement_iterations):
    #     tt_output = tt_model(
    #         spatial=tt_spatial,
    #         prompt=tt_prompt,
    #         pooled_projection=tt_pooled_projection,
    #         timestep=tt_timestep,
    #         N=spatial_sequence_length,
    #         L=prompt_sequence_length,
    #     )
    # profiler.end(f"run")
    # devices = mesh_device.get_devices()
    # ttnn.DumpDeviceProfiler(devices[0])
    # total_time = profiler.get("run")
    # avg_time = total_time / num_measurement_iterations
    # print(f" TOTAL TIME: {total_time} AVG TIME: {avg_time}\n")

    torch_output = torch.unsqueeze(torch_output, 1)
    assert_quality(torch_output, tt_output, pcc=0.997, mse=0.06, shard_dim=0, num_devices=mesh_device.get_num_devices())
