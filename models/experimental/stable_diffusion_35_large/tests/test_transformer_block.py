# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING

import pytest
import torch
import ttnn

from ..reference import SD3Transformer2DModel
from ..tt.transformer_block import TtTransformerBlock, TtTransformerBlockParameters
from ..tt.utils import assert_quality, from_torch_fast

if TYPE_CHECKING:
    from ..reference.transformer_block import TransformerBlock

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
    ("model_version", "block_index", "batch_size", "spatial_sequence_length", "prompt_sequence_length"),
    [
        ("large", 0, 2, 4096, 333),
        #        ("large", 37, 2, 4096, 333),
    ],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 716800}], indirect=True)
@pytest.mark.usefixtures("use_program_cache")
def test_transformer_block(
    *,
    mesh_device: ttnn.MeshDevice,
    model_version,
    block_index: int,
    batch_size: int,
    spatial_sequence_length: int,
    prompt_sequence_length: int,
    model_location_generator,
) -> None:
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16

    model_name = model_location_generator(
        f"stabilityai/stable-diffusion-3.5-{model_version}", model_subdir="StableDiffusion_35_Large"
    )
    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        model_name, subfolder="transformer", torch_dtype=torch_dtype
    )

    if model_version == "medium":
        embedding_dim = 1536
    else:
        embedding_dim = 2432

    torch_model: TransformerBlock = parent_torch_model.transformer_blocks[block_index]
    torch_model.eval()

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
        hidden_dim_padding = 0
        num_heads = torch_model.num_heads

    parameters = TtTransformerBlockParameters.from_torch(
        torch_model.state_dict(),
        num_heads=num_heads,
        unpadded_num_heads=torch_model.num_heads,
        hidden_dim_padding=hidden_dim_padding,
        device=mesh_device,
        dtype=ttnn_dtype,
    )

    tt_model = TtTransformerBlock(parameters, num_heads=num_heads, device=mesh_device)

    torch.manual_seed(0)
    spatial = torch.randn((batch_size, spatial_sequence_length, embedding_dim))
    prompt = torch.randn((batch_size, prompt_sequence_length, embedding_dim))
    time = torch.randn((batch_size, embedding_dim))

    ##
    spatial_padded_4d = spatial.unsqueeze(1)
    if pad_embedding_dim:
        spatial_padded_4d = torch.nn.functional.pad(
            spatial_padded_4d, pad=(0, hidden_dim_padding), mode="constant", value=0
        )
    tt_spatial = from_torch_fast(
        spatial_padded_4d, dtype=ttnn_dtype, device=mesh_device, layout=ttnn.TILE_LAYOUT, shard_dim=-1
    )

    ##
    prompt_padded_4d = prompt.unsqueeze(1)
    if pad_embedding_dim:
        prompt_padded_4d = torch.nn.functional.pad(
            prompt_padded_4d, pad=(0, hidden_dim_padding), mode="constant", value=0
        )
    tt_prompt = from_torch_fast(
        prompt_padded_4d, dtype=ttnn_dtype, device=mesh_device, layout=ttnn.TILE_LAYOUT, shard_dim=-1
    )

    ##
    if pad_embedding_dim:
        time_padded_2d = torch.nn.functional.pad(time, pad=(0, hidden_dim_padding), mode="constant", value=0)
        tt_time = from_torch_fast(
            time_padded_2d.unsqueeze(1).unsqueeze(1), dtype=ttnn_dtype, device=mesh_device, layout=ttnn.TILE_LAYOUT
        )
    else:
        tt_time = from_torch_fast(
            time.unsqueeze(1).unsqueeze(1), dtype=ttnn_dtype, device=mesh_device, layout=ttnn.TILE_LAYOUT
        )

    with torch.no_grad():
        spatial_output, prompt_output = torch_model(spatial=spatial, prompt=prompt, time_embed=time)

    tt_spatial_output_padded, tt_prompt_output_padded = tt_model(
        spatial=tt_spatial, prompt=tt_prompt, time_embed=tt_time, N=spatial_sequence_length, L=prompt_sequence_length
    )

    tt_spatial_output_padded = ttnn.to_torch(
        tt_spatial_output_padded, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)
    )
    tt_spatial_output_padded = tt_spatial_output_padded[:, :, 0:spatial_sequence_length, :embedding_dim]

    if tt_prompt_output_padded is not None:
        tt_prompt_output_padded = ttnn.to_torch(
            tt_prompt_output_padded, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)
        )
        tt_prompt_output_padded = tt_prompt_output_padded[:, :, 0:prompt_sequence_length, :embedding_dim]

    assert (prompt_output is None) == (tt_prompt_output_padded is None)
    assert_quality(
        spatial_output, tt_spatial_output_padded, pcc=0.995, shard_dim=0, num_devices=mesh_device.get_num_devices()
    )

    if prompt_output is not None and tt_prompt_output_padded is not None:
        assert_quality(
            prompt_output, tt_prompt_output_padded, pcc=0.995, shard_dim=0, num_devices=mesh_device.get_num_devices()
        )
