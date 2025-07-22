# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import pytest
import torch
import ttnn
import math

from ..reference import SD3Transformer2DModel
from ..tt.fun_transformer_block import sd_transformer_block, TtTransformerBlockParameters
from ..tt.utils import assert_quality, from_torch_fast_2d
from ..tt.parallel_config import StableDiffusionParallelManager

if TYPE_CHECKING:
    from ..reference.transformer_block import TransformerBlock

TILE_SIZE = 32


@pytest.mark.parametrize(
    (
        "model_version",
        "block_index",
        "batch_size",
        "spatial_sequence_length",
        "prompt_sequence_length",
    ),
    [
        ("large", 0, 1, 4096, 333),
        #        ("large", 37, 2, 4096, 333),
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
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 716800}], indirect=True
)
def test_transformer_block(
    *,
    mesh_device: ttnn.MeshDevice,
    model_version: str,
    block_index: int,
    batch_size: int,
    spatial_sequence_length: int,
    prompt_sequence_length: int,
    cfg: int,
    sp: int,
    tp: int,
    topology: ttnn.Topology,
    num_links: int,
    model_location_generator,
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

    model_name = model_location_generator(
        f"stabilityai/stable-diffusion-3.5-{model_version}", model_subdir="StableDiffusion_35_Large"
    )

    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        model_name,
        subfolder="transformer",
        torch_dtype=torch_dtype,
    )
    embedding_dim = 1536 if model_version == "medium" else 2432

    torch_model: TransformerBlock = parent_torch_model.transformer_blocks[block_index]
    torch_model.eval()

    ## heads padding
    assert not embedding_dim % torch_model.num_heads, "Embedding_dim % num_heads != 0"
    pad_embedding_dim = ((torch_model.num_heads) % tp_factor) != 0
    hidden_dim_padding = 0
    head_size = embedding_dim // torch_model.num_heads
    if pad_embedding_dim:
        num_heads = math.ceil(torch_model.num_heads / tp_factor) * tp_factor
        hidden_dim_padding = (num_heads * head_size) - embedding_dim
    else:
        num_heads = torch_model.num_heads

    parameters = TtTransformerBlockParameters.from_torch(
        torch_model.state_dict(),
        num_heads=num_heads,
        unpadded_num_heads=torch_model.num_heads,
        hidden_dim_padding=hidden_dim_padding,
        device=submesh,
        dtype=ttnn_dtype,
        parallel_config=parallel_manager.dit_parallel_config,
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

    spatial_shard_dims = [None, None]
    spatial_shard_dims[parallel_manager.dit_parallel_config.sequence_parallel.mesh_axis] = 2
    spatial_shard_dims[parallel_manager.dit_parallel_config.tensor_parallel.mesh_axis] = 3

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

    if pad_embedding_dim:
        time_padded = torch.nn.functional.pad(time, pad=(0, hidden_dim_padding), mode="constant", value=0)
    else:
        time_padded = time

    with torch.no_grad():
        spatial_output, prompt_output = torch_model(spatial=spatial, prompt=prompt, time_embed=time)

    # Create persistent buffers
    persistent_buffer_shape = [1, num_heads // tp_factor, spatial_padded_4d.shape[2], head_size]

    tt_spatial = from_torch_fast_2d(
        spatial_padded_4d,
        mesh_device=submesh,
        mesh_shape=parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape,
        dims=spatial_shard_dims,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
    )
    prompt_shard_dims = [None, None]
    if parallel_manager.is_tensor_parallel:
        prompt_shard_dims[parallel_manager.dit_parallel_config.tensor_parallel.mesh_axis] = 3
    tt_prompt = from_torch_fast_2d(
        prompt_padded_4d,
        mesh_device=submesh,
        mesh_shape=parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape,
        dims=prompt_shard_dims,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn_dtype,
    )

    tt_time = from_torch_fast_2d(
        time_padded.unsqueeze(1).unsqueeze(1),
        mesh_device=submesh,
        mesh_shape=parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape,
        dims=[None, None],
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn_dtype,
    )
    parallel_manager.maybe_init_persistent_buffers(
        persistent_buffer_shape, list(tt_spatial.padded_shape), list(tt_prompt.padded_shape)
    )
    for _ in range(1):
        print(f"tt_spatial: {tt_spatial.shape}")
        print(f"tt_prompt: {tt_prompt.shape}")
        print(f"tt_time: {tt_time.shape}")
        tt_spatial_output, tt_prompt_output = sd_transformer_block(
            spatial=tt_spatial,
            prompt=tt_prompt,
            time_embed=tt_time,
            parameters=parameters,
            parallel_manager=parallel_manager,
            num_heads=num_heads,
            N=spatial_sequence_length,
            L=prompt_sequence_length,
            cfg_index=0,
        )

        tt_spatial_output_torch = ttnn.to_torch(
            tt_spatial_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                submesh,
                mesh_shape=tuple(submesh.shape),
                dims=spatial_shard_dims,
            ),
        )
        tt_spatial_output_torch = tt_spatial_output_torch[:, :, 0:spatial_sequence_length, :embedding_dim]
        assert_quality(
            spatial_output, tt_spatial_output_torch, pcc=0.97, shard_dim=0, num_devices=submesh.get_num_devices()
        )

        prompt_shard_dims[
            parallel_manager.dit_parallel_config.sequence_parallel.mesh_axis
        ] = 2  # Concat replicas on sequence
        if not parallel_manager.is_tensor_parallel:
            prompt_shard_dims[parallel_manager.dit_parallel_config.tensor_parallel.mesh_axis] = 0
        if tt_prompt_output is not None:
            tt_prompt_output_torch = ttnn.to_torch(
                tt_prompt_output,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    submesh,
                    mesh_shape=tuple(submesh.shape),
                    dims=prompt_shard_dims,
                ),
            )
            tt_prompt_output_torch = tt_prompt_output_torch[:, :, 0:prompt_sequence_length, :embedding_dim]
            assert_quality(
                prompt_output, tt_prompt_output_torch, pcc=0.98, shard_dim=0, num_devices=submesh.get_num_devices()
            )
