# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest
import torch
import ttnn

from models.experimental.flux.reference.normalization import RmsNorm as RmsNormReference
from models.experimental.flux.tt.normalization import LayerNorm, LayerNormParameters, RmsNorm, RmsNormParameters
from models.experimental.flux.tt.utils import assert_quality


@pytest.mark.parametrize(
    "input_shape",
    [
        [4096, 3072],
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
@pytest.mark.parametrize("affine", [True, False], ids=["affine", "noaffine"])
def test_layer_norm(
    *,
    mesh_device: ttnn.MeshDevice,
    input_shape: list[int],
    affine: bool,
) -> None:
    batch_size, _ = mesh_device.shape
    input_shape = [batch_size, *input_shape]

    torch.manual_seed(0)

    torch_model = torch.nn.LayerNorm(input_shape[-1:], eps=1.0, elementwise_affine=affine)

    parameters = LayerNormParameters.from_torch(
        torch_model.state_dict(),
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        weight_shape=input_shape[-1:],
    )
    tt_model = LayerNorm(parameters, eps=torch_model.eps)

    torch_input_tensor = torch.randn(input_shape)

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

    composer = ttnn.ConcatMesh2dToTensor(mesh_device, tuple(mesh_device.shape), (0, -1))
    assert_quality(torch_output, tt_output, pcc=0.99990, mesh_composer=composer)


@pytest.mark.parametrize(
    "input_shape",
    [
        [2, 24, 4096, 64],
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
def test_rms_norm(
    *,
    mesh_device: ttnn.MeshDevice,
    input_shape: list[int],
) -> None:
    torch.manual_seed(0)

    torch_model = RmsNormReference(dim=input_shape[-1], eps=1.0)
    torch.nn.init.normal_(torch_model.weight)

    parameters = RmsNormParameters.from_torch(torch_model.state_dict(), device=mesh_device, dtype=ttnn.bfloat8_b)

    tt_model = RmsNorm(parameters, eps=torch_model.eps)

    torch_input_tensor = torch.randn(input_shape)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b
    )

    torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model.forward(tt_input_tensor)
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[: input_shape[0]]

    assert_quality(torch_output, tt_output_torch, pcc=0.99985)
