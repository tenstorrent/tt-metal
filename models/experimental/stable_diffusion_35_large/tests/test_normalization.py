# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import ttnn

from ..reference.normalization import RmsNorm
from ..tt.normalization import TtLayerNorm, TtLayerNormParameters, TtRmsNorm, TtRmsNormParameters
from ..tt.utils import assert_quality, to_torch


@pytest.mark.parametrize(
    "input_shape",
    [
        [4096, 3072],
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfloat8_b", "bfloat16"])
@pytest.mark.parametrize("mesh_device", [(1, 1), (1, 2), (1, 8)], indirect=True)
@pytest.mark.parametrize("affine", [True, False], ids=["affine", "noaffine"])
def test_layer_norm(
    *,
    mesh_device: ttnn.MeshDevice,
    input_shape: list[int],
    dtype: ttnn.DataType,
    affine: bool,
) -> None:
    batch_size, _ = mesh_device.shape
    input_shape = [batch_size, *input_shape]

    torch.manual_seed(0)

    torch_model = torch.nn.LayerNorm(input_shape[-1:], eps=1.0, elementwise_affine=affine)

    parameters = TtLayerNormParameters.from_torch(
        torch_model.state_dict(),
        device=mesh_device,
        dtype=dtype,
        weight_shape=input_shape[-1:],
    )
    tt_model = TtLayerNorm(parameters, eps=torch_model.eps)

    torch_input_tensor = torch.randn(input_shape)

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), (0, -1)),
    )

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)

    composer = ttnn.ConcatMesh2dToTensor(mesh_device, tuple(mesh_device.shape), (0, -1))
    tt_output_torch = to_torch(tt_output, mesh_composer=composer, fix_special_numbers=True, dtype=torch.float32)
    assert_quality(torch_output, tt_output_torch, pcc=0.99990)


@pytest.mark.parametrize(
    "input_shape",
    [
        [2, 24, 4096, 64],
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfloat8_b", "bfloat16"])
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_rms_norm(
    *,
    mesh_device: ttnn.MeshDevice,
    input_shape: list[int],
    dtype: ttnn.DataType,
) -> None:
    torch.manual_seed(0)

    torch_model = RmsNorm(dim=input_shape[-1], eps=1.0)
    torch.nn.init.normal_(torch_model.weight)

    parameters = TtRmsNormParameters.from_torch(torch_model.state_dict(), device=mesh_device, dtype=dtype)

    tt_model = TtRmsNorm(parameters, eps=torch_model.eps)

    torch_input_tensor = torch.randn(input_shape)

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype)

    torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[: input_shape[0]]

    assert_quality(torch_output, tt_output_torch, pcc=0.9998, mse=3e-4)
