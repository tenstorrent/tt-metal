# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import os
import pytest
import torch
import ttnn

from ..reference import SD3Transformer2DModel
from ..tt.transformer_block import TtTransformerBlock, TtTransformerBlockParameters
from ..tt.utils import from_torch, assert_quality

if TYPE_CHECKING:
    from ..reference.transformer_block import TransformerBlock


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
    ("model_name", "block_index", "batch_size", "spatial_sequence_length", "prompt_sequence_length"),
    [
        ("large", 0, 1, 4096, 333),
        #        ("large", 23, 1, 4096, 333),
    ],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 716800}], indirect=True)
@pytest.mark.usefixtures("use_program_cache")
def test_transformer_block(
    *,
    mesh_device: ttnn.MeshDevice,
    model_name,
    block_index: int,
    batch_size: int,
    spatial_sequence_length: int,
    prompt_sequence_length: int,
) -> None:
    mesh_device.enable_async(True)
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16

    parent_torch_model = SD3Transformer2DModel.from_pretrained(
        f"stabilityai/stable-diffusion-3.5-{model_name}", subfolder="transformer", torch_dtype=torch_dtype
    )
    if model_name == "medium":
        embedding_dim = 1536
    else:
        embedding_dim = 2432

    torch_model: TransformerBlock = parent_torch_model.transformer_blocks[block_index]
    torch_model.eval()

    parameters = TtTransformerBlockParameters.from_torch(
        torch_model.state_dict(), num_heads=torch_model.num_heads, device=mesh_device, dtype=ttnn_dtype
    )

    ## heads padding for T3K TP
    pad_40_heads = 0
    if os.environ["FAKE_DEVICE"] == "T3K" and embedding_dim == 2432:
        pad_40_heads = 1
        embedding_dim_padding = 128
        num_heads = 40
    else:
        num_heads = torch_model.num_heads

    tt_model = TtTransformerBlock(parameters, num_heads=num_heads, device=mesh_device)

    torch.manual_seed(0)
    spatial = torch.randn((batch_size, spatial_sequence_length, embedding_dim))
    prompt = torch.randn((batch_size, prompt_sequence_length, embedding_dim))
    time = torch.randn((batch_size, embedding_dim))

    TILE_SIZE = 32
    ##
    spatial_extra = spatial_sequence_length % TILE_SIZE
    if spatial_extra > 0:
        spatial_padding = TILE_SIZE - spatial_extra
    else:
        spatial_padding = 0
    spatial_padded_4D = torch.nn.functional.pad(
        spatial.unsqueeze(0), pad=(0, 0, 0, spatial_padding), mode="constant", value=0
    )
    if pad_40_heads:
        spatial_padded_4D = torch.nn.functional.pad(
            spatial_padded_4D, pad=(0, embedding_dim_padding), mode="constant", value=0
        )
    # tt_spatial = from_torch(spatial_padded_4D, dtype=ttnn_dtype, mesh_device=mesh_device, layout=ttnn.TILE_LAYOUT, shard_dim=None)
    tt_spatial = from_torch(
        spatial_padded_4D, dtype=ttnn_dtype, mesh_device=mesh_device, layout=ttnn.TILE_LAYOUT, shard_dim=-1
    )

    ##
    prompt_extra = prompt_sequence_length % TILE_SIZE
    if prompt_extra > 0:
        prompt_padding = TILE_SIZE - prompt_extra
    else:
        prompt_padding = 0
    prompt_padded_4D = torch.nn.functional.pad(
        prompt.unsqueeze(0), pad=(0, 0, 0, prompt_padding), mode="constant", value=0
    )
    if pad_40_heads:
        prompt_padded_4D = torch.nn.functional.pad(
            prompt_padded_4D, pad=(0, embedding_dim_padding), mode="constant", value=0
        )
    # tt_prompt = from_torch(prompt_padded_4D, dtype=ttnn_dtype, mesh_device=mesh_device, layout=ttnn.TILE_LAYOUT, shard_dim=None)
    tt_prompt = from_torch(
        prompt_padded_4D, dtype=ttnn_dtype, mesh_device=mesh_device, layout=ttnn.TILE_LAYOUT, shard_dim=-1
    )

    ##
    if pad_40_heads:
        time_padded_2D = torch.nn.functional.pad(time, pad=(0, embedding_dim_padding), mode="constant", value=0)
        tt_time = from_torch(
            time_padded_2D.unsqueeze(1), dtype=ttnn_dtype, mesh_device=mesh_device, layout=ttnn.TILE_LAYOUT
        )
    else:
        tt_time = from_torch(time.unsqueeze(1), dtype=ttnn_dtype, mesh_device=mesh_device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        spatial_output, prompt_output = torch_model(spatial=spatial, prompt=prompt, time_embed=time)

    # tt_spatial = allocate_tensor_on_device_like(tt_spatial_host, device=device)
    # tt_prompt = allocate_tensor_on_device_like(tt_prompt_host, device=device)
    # tt_time = allocate_tensor_on_device_like(tt_time_host, device=device)

    # ttnn.copy_host_to_device_tensor(tt_spatial_host, tt_spatial)
    # ttnn.copy_host_to_device_tensor(tt_prompt_host, tt_prompt)
    # ttnn.copy_host_to_device_tensor(tt_time_host, tt_time)

    tt_spatial_output_padded, tt_prompt_output_padded = tt_model(
        spatial=tt_spatial, prompt=tt_prompt, time_embed=tt_time
    )
    tt_spatial_output_padded = ttnn.to_torch(
        tt_spatial_output_padded, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)
    )
    tt_spatial_output_padded = tt_spatial_output_padded[:, :, 0:spatial_sequence_length, :embedding_dim]
    # tt_spatial_output_padded = tt_spatial_output_padded[:, :, 0:spatial_sequence_length, :embedding_dim]

    tt_prompt_output_padded = ttnn.to_torch(
        tt_prompt_output_padded, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)
    )
    tt_prompt_output_padded = tt_prompt_output_padded[:, :, 0:prompt_sequence_length, :embedding_dim]

    assert (prompt_output is None) == (tt_prompt_output_padded is None)
    assert_quality(
        spatial_output, tt_spatial_output_padded, pcc=0.995, shard_dim=0, num_devices=mesh_device.get_num_devices()
    )
    # assert_quality(spatial_output, tt_spatial_output_padded, pcc=0.995, shard_dim=0, num_devices=mesh_device.get_num_devices())

    if prompt_output is not None and tt_prompt_output_padded is not None:
        assert_quality(
            prompt_output, tt_prompt_output_padded, pcc=0.995, shard_dim=0, num_devices=mesh_device.get_num_devices()
        )
