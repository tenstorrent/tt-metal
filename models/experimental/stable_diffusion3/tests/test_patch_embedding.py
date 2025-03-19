# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import os
import pytest
import torch
import ttnn

from ..reference import SD3Transformer2DModel
from ..tt.patch_embedding import TtPatchEmbed, TtPatchEmbedParameters
from ..tt.utils import assert_quality, to_torch

if TYPE_CHECKING:
    from ..reference.patch_embedding import PatchEmbed


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
    dtype = ttnn.bfloat16

    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        f"stabilityai/stable-diffusion-3.5-{model_name}", subfolder="transformer", torch_dtype=torch.bfloat16
    )
    if model_name == "medium":
        embedding_dim = 1536
    else:
        embedding_dim = 2432

    torch_model: PatchEmbed = parent_torch_model.pos_embed
    torch_model.eval()

    parameters = TtPatchEmbedParameters.from_torch(
        torch_model.state_dict(), device=mesh_device, out_channels=embedding_dim
    )
    tt_model = TtPatchEmbed(parameters, mesh_device=mesh_device)

    torch_input_tensor = torch.randn((batch_size, 16, 128, 128), dtype=torch.bfloat16)

    torch_output = torch_model(torch_input_tensor)

    """
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor.permute([0, 2, 3, 1]),  # BCYX -> BYXC
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
    )
    """

    ## Pre-processing for the ttnn.fold
    torch_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))  # BCYX -> BYXC
    batch_size, img_h, img_w, img_c = torch_input_tensor.shape  # permuted input NHWC
    patch_size = 2
    torch_input_tensor = torch_input_tensor.reshape(batch_size, img_h, img_w // patch_size, patch_size, img_c)
    torch_input_tensor = torch_input_tensor.reshape(batch_size, img_h, img_w // patch_size, patch_size * img_c)
    N, H, W, C = torch_input_tensor.shape
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 7),
            ),
        }
    )
    n_cores = 64
    shard_spec = ttnn.ShardSpec(shard_grid, [N * H * W // n_cores, C], ttnn.ShardOrientation.ROW_MAJOR)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        ),
    )

    tt_output = tt_model(tt_input_tensor)

    tt_output_torch = to_torch(tt_output, mesh_device=mesh_device, dtype=dtype, shard_dim=0).squeeze(1)[
        :batch_size, :, :embedding_dim
    ]

    assert_quality(torch_output, tt_output_torch, pcc=0.999_990, shard_dim=0, num_devices=mesh_device.get_num_devices())
