# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import os
import pytest
import torch
import ttnn

from ..reference import SD3Transformer2DModel
from ..tt.fun_patch_embedding import sd_patch_embed, TtPatchEmbedParameters
from ..tt.utils import assert_quality, to_torch
from ..tt.parallel_config import create_dit_parallel_config, ParallelConfig

if TYPE_CHECKING:
    from ..reference.patch_embedding import PatchEmbed

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
    ("model_name", "batch_size"),
    [
        ("large", 2),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.usefixtures("use_program_cache")
def test_patch_embedding(
    *,
    mesh_device: ttnn.MeshDevice,
    model_name,
    batch_size: int,
) -> None:
    mesh_shape = tuple(mesh_device.shape)
    cfg_parallel = ParallelConfig(mesh_shape=mesh_shape, factor=1, mesh_axis=0)
    tensor_parallel = ParallelConfig(mesh_shape=(mesh_shape[0], 1), factor=mesh_shape[1], mesh_axis=1)
    dit_parallel_config = create_dit_parallel_config(
        mesh_shape=mesh_shape, cfg_parallel=cfg_parallel, tensor_parallel=tensor_parallel
    )
    dtype = ttnn.bfloat16

    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        f"stabilityai/stable-diffusion-3.5-{model_name}", subfolder="transformer", torch_dtype=torch.bfloat16
    )
    if model_name == "medium":
        embedding_dim = 1536
    else:
        embedding_dim = 2432

    num_devices = mesh_device.get_num_devices()
    pad_embedding_dim = False
    if os.environ["MESH_DEVICE"] == "T3K" and embedding_dim == 2432:
        pad_embedding_dim = True
        hidden_dim_padding = (
            ((embedding_dim // num_devices // TILE_SIZE) + 1) * TILE_SIZE
        ) * num_devices - embedding_dim
    else:
        hidden_dim_padding = 0

    torch_model: PatchEmbed = parent_torch_model.pos_embed
    torch_model.eval()

    parameters = TtPatchEmbedParameters.from_torch(
        torch_model.state_dict(),
        device=mesh_device,
        hidden_dim_padding=hidden_dim_padding,
        out_channels=embedding_dim,
        parallel_config=dit_parallel_config,
    )

    torch_input_tensor = torch.randn((batch_size, 16, 128, 128), dtype=torch.bfloat16)

    torch_output = torch_model(torch_input_tensor)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor.permute([0, 2, 3, 1]),  # BCYX -> BYXC
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
    )

    tt_output = sd_patch_embed(tt_input_tensor, parameters, parallel_config=dit_parallel_config)

    tt_output_torch = to_torch(tt_output, mesh_device=mesh_device, dtype=dtype, shard_dim=-1).squeeze(1)[
        :batch_size, :, :embedding_dim
    ]

    assert_quality(torch_output, tt_output_torch, pcc=0.999_990, shard_dim=0, num_devices=mesh_device.get_num_devices())
