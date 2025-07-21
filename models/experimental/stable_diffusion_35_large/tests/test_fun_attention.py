# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import pytest
import torch
import ttnn
import math

from ..reference import SD3Transformer2DModel
from ..tt.fun_attention import sd_joint_attention, TtAttentionParameters
from ..tt.utils import assert_quality, from_torch_fast_2d
from ..tt.parallel_config import StableDiffusionParallelManager

if TYPE_CHECKING:
    from ..reference.attention import Attention

TILE_SIZE = 32


@pytest.mark.parametrize(
    (
        "model_name",
        "block_index",
        "batch_size",
        "spatial_sequence_length",
        "prompt_sequence_length",
    ),
    [
        # ("medium", 0, 1, 4096, 333),
        # ("medium", 23, 1, 4096, 333),
        ("large", 0, 1, 4096, 333),
        # ("large", 23, 1, 4096, 333),
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
@pytest.mark.parametrize("joint_attention", [True])
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 517120}], indirect=True
)
def test_attention(
    *,
    mesh_device: ttnn.MeshDevice,
    model_name: str,
    block_index: int,
    batch_size: int,
    spatial_sequence_length: int,
    prompt_sequence_length: int,
    joint_attention: bool,
    cfg: tuple[int, int],
    sp: tuple[int, int],
    tp: tuple[int, int],
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

    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        f"stabilityai/stable-diffusion-3.5-{model_name}", subfolder="transformer", torch_dtype=torch_dtype
    )
    embedding_dim = 1536 if model_name == "medium" else 2432

    torch_model: Attention = parent_torch_model.transformer_blocks[block_index].attn
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

    parameters = TtAttentionParameters.from_torch(
        torch_model.state_dict(),
        num_heads=num_heads,
        unpadded_num_heads=torch_model.num_heads,
        hidden_dim_padding=hidden_dim_padding,
        device=submesh,
        dtype=ttnn_dtype,
        parallel_config=parallel_manager.dit_parallel_config,
    )

    torch.manual_seed(0)
    spatial = torch.randn((batch_size, spatial_sequence_length, embedding_dim), dtype=torch_dtype)
    prompt = (
        torch.randn((batch_size, prompt_sequence_length, embedding_dim), dtype=torch_dtype) if joint_attention else None
    )

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
    shard_spatial_dims = [None, None]
    shard_spatial_dims[parallel_manager.dit_parallel_config.sequence_parallel.mesh_axis] = 2
    tt_spatial = from_torch_fast_2d(
        spatial_padded_4d,
        mesh_device=submesh,
        mesh_shape=parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape,
        dims=shard_spatial_dims,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn_dtype,
    )

    if joint_attention:
        prompt_padded_4d = prompt.unsqueeze(1)
        if pad_embedding_dim:
            prompt_padded_4d = torch.nn.functional.pad(
                prompt_padded_4d, pad=(0, hidden_dim_padding), mode="constant", value=0
            )
        if prompt_padded_4d.shape[2] % TILE_SIZE:
            prompt_padded_4d = torch.nn.functional.pad(
                prompt_padded_4d,
                pad=(0, 0, 0, TILE_SIZE - (prompt_padded_4d.shape[2] % TILE_SIZE)),
                mode="constant",
                value=0,
            )
        tt_prompt = from_torch_fast_2d(
            prompt_padded_4d,
            mesh_device=submesh,
            mesh_shape=parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape,
            dims=[None, None],
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn_dtype,
        )
    else:
        tt_prompt = None

    with torch.no_grad():
        spatial_output, prompt_output = torch_model(spatial=spatial, prompt=prompt)

    # Create persistent buffers
    persistent_buffer_shape = [1, num_heads // tp_factor, spatial_padded_4d.shape[2], head_size]
    sharded_spatial_shape = list(tt_spatial.padded_shape)
    sharded_spatial_shape[3] //= tp_factor
    sharded_prompt_shape = list(tt_prompt.padded_shape)
    sharded_prompt_shape[3] //= tp_factor
    parallel_manager.maybe_init_persistent_buffers(persistent_buffer_shape, sharded_spatial_shape, sharded_prompt_shape)

    # if joint_attention:
    tt_spatial_output, tt_prompt_output = sd_joint_attention(
        spatial=tt_spatial,
        prompt=tt_prompt,
        parameters=parameters,
        parallel_manager=parallel_manager,
        num_heads=num_heads,
        N=spatial_sequence_length,
        L=prompt_sequence_length,
        cfg_index=0,
    )

    ttnn.synchronize_device(submesh)

    gather_spatial_dims = [None, None]
    gather_spatial_dims[parallel_manager.dit_parallel_config.sequence_parallel.mesh_axis] = 2
    gather_spatial_dims[parallel_manager.dit_parallel_config.tensor_parallel.mesh_axis] = 3
    tt_spatial_output_torch = ttnn.to_torch(
        tt_spatial_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(submesh.shape), dims=gather_spatial_dims),
    )
    tt_spatial_output_torch = tt_spatial_output_torch[:, :, 0:spatial_sequence_length, :embedding_dim]
    assert_quality(
        spatial_output, tt_spatial_output_torch, pcc=0.990, shard_dim=0, num_devices=submesh.get_num_devices()
    )

    if joint_attention:
        tt_prompt_output_torch = ttnn.to_torch(
            tt_prompt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device,
                mesh_shape=tuple(submesh.shape),
                dims=gather_spatial_dims,
            ),
        )
        tt_prompt_output_torch = tt_prompt_output_torch[:, :, 0:prompt_sequence_length, :embedding_dim]
        assert_quality(
            prompt_output, tt_prompt_output_torch, pcc=0.990, shard_dim=0, num_devices=submesh.get_num_devices()
        )
