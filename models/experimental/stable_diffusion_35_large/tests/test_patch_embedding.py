# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
import torch
import ttnn

from ..reference import SD3Transformer2DModel
from ..tt.patch_embedding import TtPatchEmbed, TtPatchEmbedParameters
from ..tt.utils import assert_quality

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
    model_location_generator,
) -> None:
    dtype = ttnn.bfloat16

    model_version = model_name
    model_name = model_location_generator(
        f"stabilityai/stable-diffusion-3.5-{model_version}", model_subdir="StableDiffusion_35_Large"
    )
    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        model_name, subfolder="transformer", torch_dtype=torch.bfloat16
    )

    if model_version == "medium":
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
        torch_model.state_dict(), device=mesh_device, hidden_dim_padding=hidden_dim_padding, out_channels=embedding_dim
    )
    tt_model = TtPatchEmbed(parameters, mesh_device=mesh_device)

    torch_input_tensor = torch.randn((batch_size, 16, 128, 128), dtype=torch.bfloat16)

    torch_output = torch_model(torch_input_tensor)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor.permute([0, 2, 3, 1]),  # BCYX -> BYXC
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
    )

    tt_output = tt_model(tt_input_tensor)

    tt_output = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    tt_output = tt_output[:, :, :embedding_dim]

    assert_quality(torch_output, tt_output, pcc=0.999_990, shard_dim=-1)
