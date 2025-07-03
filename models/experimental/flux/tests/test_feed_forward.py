# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest
import torch
import ttnn

from models.experimental.flux.reference.feed_forward import FeedForward as FeedForwardReference
from models.experimental.flux.tt.feed_forward import FeedForward, FeedForwardParameters
from models.experimental.flux.tt.utils import assert_quality


@pytest.mark.parametrize(
    ("batch_size", "input_dim", "output_dim"),
    [
        (4096, 3072, 3072),
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE") or "N300", len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize("linear_on_host", [False, True])
def test_feed_forward(
    *,
    mesh_device: ttnn.MeshDevice,
    batch_size: int,
    input_dim: int,
    output_dim: int,
    linear_on_host: bool,
) -> None:
    torch.manual_seed(0)

    torch_model = FeedForwardReference(dim=input_dim, dim_out=output_dim)
    torch_model.eval()

    parameters = FeedForwardParameters.from_torch(
        torch_model.state_dict(),
        device=mesh_device,
        linear_on_host=linear_on_host,
        dtype=ttnn.bfloat8_b,
        mesh_sharded_input=True,
    )
    tt_model = FeedForward(parameters)

    torch_input_tensor = torch.randn((batch_size, input_dim))

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), (0, -1)),
    )

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model.forward(tt_input_tensor)

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, tuple(mesh_device.shape), (0, -1)),
    )

    assert_quality(torch_output, tt_output_torch, pcc=0.99907, mse=0.00025)
