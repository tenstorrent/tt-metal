# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest
import torch
import ttnn

from ..tt.linear import Linear, LinearParameters
from ..tt.utils import assert_quality


@pytest.mark.parametrize(
    ("batch_size", "input_dim", "output_dim"),
    [
        (32, 1536, 2048),
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_sharding_dim", [0, 1, None], ids=["in_sharding", "out_sharding", "no_sharding"])
@pytest.mark.parametrize("on_host", [False, True], ids=["host", "device"])
def test_linear(
    *,
    mesh_device: ttnn.MeshDevice,
    batch_size: int,
    input_dim: int,
    output_dim: int,
    mesh_sharding_dim: int | None,
    on_host: bool,
) -> None:
    torch.manual_seed(0)

    torch_model = torch.nn.Linear(input_dim, output_dim)
    torch_model.eval()

    parameters = LinearParameters.from_torch(
        torch_model.state_dict(),
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        mesh_sharding_dim=mesh_sharding_dim,
        on_host=on_host,
    )
    tt_model = Linear(parameters)

    torch_input_tensor = torch.randn((batch_size, input_dim))

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, -1)
        if mesh_sharding_dim == 0
        else ttnn.ReplicateTensorToMesh(mesh_device),
    )

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model.forward(tt_input_tensor)
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1),
    )

    if mesh_sharding_dim is None:
        tt_output_torch = tt_output_torch[..., :output_dim]

    assert_quality(torch_output, tt_output_torch, pcc=0.99967)
