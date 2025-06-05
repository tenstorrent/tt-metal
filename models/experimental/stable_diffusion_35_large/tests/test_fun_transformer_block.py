# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING

import pytest
import torch
import ttnn
import math

from ..reference import SD3Transformer2DModel
from ..tt.fun_transformer_block import sd_transformer_block, TtTransformerBlockParameters
from ..tt.utils import assert_quality, from_torch_fast_2d, create_global_semaphores, initialize_sd_parallel_config

if TYPE_CHECKING:
    from ..reference.transformer_block import TransformerBlock

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
        "block_index",
        "batch_size",
        "spatial_sequence_length",
        "prompt_sequence_length",
        "cfg_factor",
        "sp_factor",
        "tp_factor",
        "topology",
    ),
    [
        ("large", 0, 1, 4096, 333, 1, 2, 4, ttnn.Topology.Linear),
        #        ("large", 37, 2, 4096, 333),
    ],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 716800}], indirect=True
)
@pytest.mark.usefixtures("use_program_cache")
def test_transformer_block(
    *,
    mesh_device: ttnn.MeshDevice,
    model_name,
    block_index: int,
    batch_size: int,
    spatial_sequence_length: int,
    prompt_sequence_length: int,
    cfg_factor: int,
    sp_factor: int,
    tp_factor: int,
    topology: ttnn.Topology,
) -> None:
    mesh_shape = tuple(mesh_device.shape)
    dit_parallel_config = initialize_sd_parallel_config(mesh_shape, cfg_factor, sp_factor, tp_factor, topology)
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )

    # create global semaphore handles
    num_devices = mesh_device.get_num_devices()
    ag_ccl_semaphore_handle = create_global_semaphores(mesh_device, num_devices, ccl_sub_device_crs, 0)
    rs_from_ccl_semaphore_handle = create_global_semaphores(mesh_device, num_devices, ccl_sub_device_crs, 0)
    rs_to_ccl_semaphore_handle = create_global_semaphores(mesh_device, num_devices, ccl_sub_device_crs, 0)

    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        f"stabilityai/stable-diffusion-3.5-{model_name}", subfolder="transformer", torch_dtype=torch_dtype
    )
    embedding_dim = 1536 if model_name == "medium" else 2432

    torch_model: TransformerBlock = parent_torch_model.transformer_blocks[block_index]
    torch_model.eval()

    ## heads padding
    assert not embedding_dim % torch_model.num_heads, "Embedding_dim % num_heads != 0"
    pad_embedding_dim = (bool)(torch_model.num_heads) % tp_factor
    if pad_embedding_dim:
        head_size = embedding_dim // torch_model.num_heads
        num_heads = math.ceil(torch_model.num_heads / tp_factor) * tp_factor
        hidden_dim_padding = (num_heads * head_size) - embedding_dim
    else:
        num_heads = torch_model.num_heads

    parameters = TtTransformerBlockParameters.from_torch(
        torch_model.state_dict(),
        num_heads=num_heads,
        unpadded_num_heads=torch_model.num_heads,
        hidden_dim_padding=hidden_dim_padding,
        device=mesh_device,
        dtype=ttnn_dtype,
        parallel_config=dit_parallel_config,
    )

    torch.manual_seed(0)
    spatial = torch.randn((batch_size, spatial_sequence_length, embedding_dim))
    prompt = torch.randn((batch_size, prompt_sequence_length, embedding_dim))
    time = torch.randn((batch_size, embedding_dim))

    spatial_padded_4d = spatial.unsqueeze(1)
    if pad_embedding_dim:
        spatial_padded_4d = torch.nn.functional.pad(
            spatial_padded_4d, pad=(0, hidden_dim_padding), mode="constant", value=0
        )
    if spatial_padded_4d.shape[2] % TILE_SIZE:
        spatial_padded_4d = torch.nn.functional.pad(
            spatial_padded_4d,
            pad=(0, 0, 0, TILE_SIZE - (spatial_padded_4d.shape[2] % TILE_SIZE)),
            mode="constant",
            value=0,
        )
    tt_spatial = from_torch_fast_2d(
        spatial_padded_4d,
        mesh_device=mesh_device,
        mesh_shape=dit_parallel_config.cfg_parallel.mesh_shape,
        dims=[dit_parallel_config.sequence_parallel.mesh_axis + 2, dit_parallel_config.tensor_parallel.mesh_axis + 2],
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
    )

    prompt_padded_4d = prompt.unsqueeze(1)
    if pad_embedding_dim:
        prompt_padded_4d = torch.nn.functional.pad(
            prompt_padded_4d, pad=(0, hidden_dim_padding), mode="constant", value=0
        )
    # if prompt_padded_4d.shape[2] % TILE_SIZE:
    #     prompt_padded_4d = torch.nn.functional.pad(
    #         prompt_padded_4d,
    #         pad=(0, 0, 0, TILE_SIZE - (prompt_padded_4d.shape[2] % TILE_SIZE)),
    #         mode="constant",
    #         value=0,
    #     )
    tt_prompt = from_torch_fast_2d(
        prompt_padded_4d,
        mesh_device=mesh_device,
        mesh_shape=dit_parallel_config.cfg_parallel.mesh_shape,
        dims=[None, dit_parallel_config.tensor_parallel.mesh_axis + 2],
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn_dtype,
    )

    if pad_embedding_dim:
        time_padded_2d = torch.nn.functional.pad(time, pad=(0, hidden_dim_padding), mode="constant", value=0)
        tt_time = from_torch_fast_2d(
            time_padded_2d.unsqueeze(1).unsqueeze(1),
            mesh_device=mesh_device,
            mesh_shape=dit_parallel_config.cfg_parallel.mesh_shape,
            dims=[None, None],
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn_dtype,
        )
    else:
        tt_time = from_torch_fast_2d(
            time.unsqueeze(1).unsqueeze(1),
            mesh_device=mesh_device,
            mesh_shape=dit_parallel_config.cfg_parallel.mesh_shape,
            dims=[None, None],
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn_dtype,
        )

    with torch.no_grad():
        spatial_output, prompt_output = torch_model(spatial=spatial, prompt=prompt, time_embed=time)

    tt_spatial_output, tt_prompt_output = sd_transformer_block(
        spatial=tt_spatial,
        prompt=tt_prompt,
        time_embed=tt_time,
        parameters=parameters,
        parallel_config=dit_parallel_config,
        num_heads=num_heads,
        N=spatial_sequence_length,
        L=prompt_sequence_length,
        ag_global_semaphore=ag_ccl_semaphore_handle,
        rs_from_global_semaphore=rs_from_ccl_semaphore_handle,
        rs_to_global_semaphore=rs_to_ccl_semaphore_handle,
    )

    ttnn.synchronize_device(mesh_device)

    # num_measurement_iterations=38
    # profiler.clear()
    # profiler.start(f"run")
    # for i in range(num_measurement_iterations):
    #     tt_spatial_output_padded, tt_prompt_output_padded = tt_model(
    #         spatial=tt_spatial, prompt=tt_prompt, time_embed=tt_time, N=spatial_sequence_length, L=prompt_sequence_length
    #     )
    # profiler.end(f"run")
    # devices = mesh_device.get_devices()
    # ttnn.DumpDeviceProfiler(devices[0])
    # total_time = profiler.get("run")
    # avg_time = total_time / num_measurement_iterations
    # print(f" TOTAL TIME: {total_time} AVG TIME: {avg_time}\n")

    tt_spatial_output_torch = ttnn.to_torch(
        tt_spatial_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device,
            mesh_shape=tuple(mesh_device.shape),
            dims=[
                dit_parallel_config.sequence_parallel.mesh_axis + 2,
                dit_parallel_config.tensor_parallel.mesh_axis + 2,
            ],
        ),
    )
    tt_spatial_output_torch = tt_spatial_output_torch[:, :, 0:spatial_sequence_length, :embedding_dim]
    assert_quality(
        spatial_output, tt_spatial_output_torch, pcc=0.995, shard_dim=0, num_devices=mesh_device.get_num_devices()
    )

    if tt_prompt_output is not None:
        tt_prompt_output_torch = ttnn.to_torch(
            tt_prompt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device,
                mesh_shape=tuple(mesh_device.shape),
                dims=[
                    dit_parallel_config.sequence_parallel.mesh_axis + 2,
                    dit_parallel_config.tensor_parallel.mesh_axis + 2,
                ],
            ),
        )
        tt_prompt_output_torch = tt_prompt_output_torch[:, :, 0:prompt_sequence_length, :embedding_dim]
        assert_quality(
            prompt_output, tt_prompt_output_torch, pcc=0.995, shard_dim=0, num_devices=mesh_device.get_num_devices()
        )
