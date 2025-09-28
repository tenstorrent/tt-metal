# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
import ttnn
import math

from ..reference import SD3Transformer2DModel
from ..tt.fun_patch_embedding import sd_patch_embed, TtPatchEmbedParameters
from ..tt.utils import assert_quality, from_torch_fast_2d
from ..tt.parallel_config import StableDiffusionParallelManager

if TYPE_CHECKING:
    from ..reference.patch_embedding import PatchEmbed

TILE_SIZE = 32


@pytest.mark.parametrize(
    (
        "model_version",
        "batch_size",
        "in_channels",
        "height",
        "width",
    ),
    [
        ("large", 1, 16, 128, 128),
        ("large", 2, 16, 128, 128),
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
        [(2, 4), (1, 0), (2, 0), (4, 1), ttnn.Topology.Linear, 1],
        [(4, 8), (2, 1), (4, 0), (4, 1), ttnn.Topology.Linear, 3],
    ],
    ids=[
        "t3k_cfg1_sp2_tp4",
        "tg_cfg2_sp4_tp4",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_patch_embedding(
    *,
    mesh_device: ttnn.MeshDevice,
    model_version,
    batch_size: int,
    in_channels: int,
    height: int,
    width: int,
    cfg: tuple[int, int],
    sp: tuple[int, int],
    tp: tuple[int, int],
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
        model_name, subfolder="transformer", torch_dtype=torch_dtype
    )
    embedding_dim = 1536 if model_version == "medium" else 2432

    torch_model: PatchEmbed = parent_torch_model.pos_embed
    torch_model.eval()

    ## heads padding
    assert not embedding_dim % parent_torch_model.transformer_blocks[0].num_heads, "Embedding_dim % num_heads != 0"
    pad_embedding_dim = (bool)(parent_torch_model.transformer_blocks[0].num_heads) % tp_factor
    if pad_embedding_dim:
        head_size = embedding_dim // parent_torch_model.transformer_blocks[0].num_heads
        num_heads = math.ceil(parent_torch_model.transformer_blocks[0].num_heads / tp_factor) * tp_factor
        hidden_dim_padding = (num_heads * head_size) - embedding_dim
    else:
        num_heads = parent_torch_model.transformer_blocks[0].num_heads

    parameters = TtPatchEmbedParameters.from_torch(
        torch_model.state_dict(),
        device=submesh,
        hidden_dim_padding=hidden_dim_padding,
        out_channels=embedding_dim,
        parallel_config=parallel_manager.dit_parallel_config,
        dtype=ttnn_dtype,
        height=height,
        width=width,
    )

    torch_input_tensor = torch.randn((batch_size, in_channels, height, width), dtype=torch_dtype)

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    seq_parallel_shard_dim = 1  # 1 is height
    dims = [None, None]
    dims[parallel_manager.dit_parallel_config.sequence_parallel.mesh_axis] = seq_parallel_shard_dim
    tt_input_tensor = from_torch_fast_2d(
        torch_input_tensor.permute([0, 2, 3, 1]),  # BCYX -> BYXC
        mesh_device=submesh,
        mesh_shape=parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape,
        dims=dims,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
    )

    concat_dims = [None, None]
    concat_dims[parallel_manager.dit_parallel_config.sequence_parallel.mesh_axis] = seq_parallel_shard_dim
    concat_dims[parallel_manager.dit_parallel_config.tensor_parallel.mesh_axis] = 2
    tt_output = sd_patch_embed(tt_input_tensor, parameters, parallel_manager=parallel_manager)
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            submesh,
            mesh_shape=tuple(submesh.shape),
            dims=concat_dims,
        ),
    )
    tt_output_torch = tt_output_torch.squeeze(1)[:batch_size, :, :embedding_dim]

    assert_quality(torch_output, tt_output_torch, pcc=0.999_990)
