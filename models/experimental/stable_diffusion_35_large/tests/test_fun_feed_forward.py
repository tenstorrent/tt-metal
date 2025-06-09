# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import ttnn
import math

from ..reference.feed_forward import FeedForward
from ..tt.fun_feed_forward import TtFeedForwardParameters, sd_feed_forward
from ..tt.utils import assert_quality, from_torch_fast_2d
from ..tt.parallel_config import StableDiffusionParallelManager

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
        "batch_size",
        "embedding_dim",
        "num_heads",
        "sequence_length",
        "cfg_factor",
        "sp_factor",
        "tp_factor",
        "topology",
        "shard_sequence",
    ),
    [
        (1, 2432, 38, 4096, 1, 2, 4, ttnn.Topology.Linear, True),
        (1, 2432, 38, 333, 1, 2, 4, ttnn.Topology.Linear, False),
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.usefixtures("use_program_cache")
def test_feed_forward(
    *,
    mesh_device: ttnn.MeshDevice,
    batch_size: int,
    embedding_dim: int,
    num_heads: int,
    sequence_length: int,
    cfg_factor: int,
    sp_factor: int,
    tp_factor: int,
    topology: ttnn.Topology,
    shard_sequence: bool,
) -> None:
    parallel_manager = StableDiffusionParallelManager(
        mesh_device, cfg_factor, sp_factor, tp_factor, sp_factor, tp_factor, topology
    )
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16

    torch_model = FeedForward(dim=embedding_dim, dim_out=embedding_dim).to(dtype=torch_dtype)
    torch_model.eval()

    ## heads padding
    pad_embedding_dim = (bool)(num_heads) % tp_factor
    if pad_embedding_dim:
        head_size = embedding_dim // num_heads
        num_heads = math.ceil(num_heads / tp_factor) * tp_factor
        hidden_dim_padding = (num_heads * head_size) - embedding_dim
    else:
        hidden_dim_padding = 0

    parameters = TtFeedForwardParameters.from_torch(
        torch_model.state_dict(),
        dtype=ttnn_dtype,
        device=mesh_device,
        hidden_dim_padding=hidden_dim_padding,
        parallel_config=parallel_manager.dit_parallel_config,
    )

    torch.manual_seed(0)
    torch_input_tensor = torch.randn((batch_size, sequence_length, embedding_dim), dtype=torch_dtype)

    input_padded_4d = torch_input_tensor.unsqueeze(1)
    if pad_embedding_dim:
        input_padded_4d = torch.nn.functional.pad(
            input_padded_4d, pad=(0, hidden_dim_padding), mode="constant", value=0
        )
    if input_padded_4d.shape[2] % TILE_SIZE:
        input_padded_4d = torch.nn.functional.pad(
            input_padded_4d,
            pad=(0, 0, 0, TILE_SIZE - (input_padded_4d.shape[2] % TILE_SIZE)),
            mode="constant",
            value=0,
        )
    tt_input_tensor = from_torch_fast_2d(
        input_padded_4d,
        mesh_device=mesh_device,
        mesh_shape=parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape,
        dims=[parallel_manager.dit_parallel_config.sequence_parallel.mesh_axis + 2 if shard_sequence else None, None],
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
    )

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = sd_feed_forward(
        tt_input_tensor,
        parameters,
        parallel_manager=parallel_manager,
        cfg_index=0,
    )
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device,
            mesh_shape=tuple(mesh_device.shape),
            dims=[
                parallel_manager.dit_parallel_config.sequence_parallel.mesh_axis + 2,
                parallel_manager.dit_parallel_config.tensor_parallel.mesh_axis + 2,
            ],
        ),
    )
    tt_output_torch = tt_output_torch[:, :, 0:sequence_length, :embedding_dim]

    assert_quality(torch_output, tt_output_torch, pcc=0.998_800, shard_dim=-1)
