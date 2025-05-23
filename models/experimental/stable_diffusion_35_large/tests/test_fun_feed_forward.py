# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import ttnn

from ..reference.feed_forward import FeedForward
from ..tt.fun_feed_forward import TtFeedForwardParameters, sd_feed_forward
from ..tt.utils import assert_quality, from_torch_fast
from ..tt.parallel_config import ParallelConfig, create_dit_parallel_config


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
    ("batch_size", "input_dim", "output_dim"),
    [
        (32, 128, 256),
    ],
)
@pytest.mark.usefixtures("use_program_cache")
def test_feed_forward(
    *,
    mesh_device: ttnn.MeshDevice,
    batch_size: int,
    input_dim: int,
    output_dim: int,
) -> None:
    mesh_shape = tuple(mesh_device.shape)
    cfg_parallel = ParallelConfig(mesh_shape=mesh_shape, factor=1, mesh_axis=0)
    tensor_parallel = ParallelConfig(mesh_shape=(mesh_shape[0], 1), factor=mesh_shape[1], mesh_axis=1)
    dit_parallel_config = create_dit_parallel_config(
        mesh_shape=mesh_shape, cfg_parallel=cfg_parallel, tensor_parallel=tensor_parallel
    )

    dtype = torch.bfloat16

    torch_model = FeedForward(dim=input_dim, dim_out=output_dim).to(dtype=dtype)
    torch_model.eval()

    parameters = TtFeedForwardParameters.from_torch(
        torch_model.state_dict(), device=mesh_device, parallel_config=dit_parallel_config, dtype=ttnn.bfloat8_b
    )

    torch_input_tensor = torch.randn((batch_size, 1, 1, input_dim), dtype=dtype)

    tt_input_tensor = from_torch_fast(torch_input_tensor, device=mesh_device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = sd_feed_forward(tt_input_tensor, parameters, parallel_config=dit_parallel_config)

    assert_quality(torch_output, tt_output, pcc=0.998_800, shard_dim=-1)
