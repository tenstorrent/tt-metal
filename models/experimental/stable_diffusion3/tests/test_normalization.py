# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

from ..reference.normalization import RmsNorm
from ..tt.normalization import TtLayerNorm, TtLayerNormParameters, TtRmsNorm, TtRmsNormParameters


@pytest.mark.parametrize(
    "input_shape",
    [
        [2, 24, 4096, 64],
    ],
)
@pytest.mark.usefixtures("use_program_cache")
def test_layer_norm(
    *,
    device: ttnn.Device,
    input_shape: list[int],
) -> None:
    dtype = torch.bfloat16

    torch_model = torch.nn.LayerNorm(input_shape[-1:], eps=1.0).to(dtype=dtype)

    parameters = TtLayerNormParameters.from_torch(torch_model.state_dict(), device=device, dtype=ttnn.bfloat8_b)
    tt_model = TtLayerNorm(parameters, eps=torch_model.eps)

    torch_input_tensor = torch.randn(input_shape, dtype=dtype)

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.TILE_LAYOUT)

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)
    tt_output_torch = ttnn.to_torch(tt_output)

    mse = torch.nn.functional.mse_loss(
        torch_output.to(dtype=torch.float32),
        tt_output_torch.to(dtype=torch.float32),
    ).item()
    logger.info(f"mse: {mse:.6f}")
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999_950)


@pytest.mark.parametrize(
    "input_shape",
    [
        [2, 24, 4096, 64],
    ],
)
@pytest.mark.usefixtures("use_program_cache")
def test_rms_norm(
    *,
    device: ttnn.Device,
    input_shape: list[int],
) -> None:
    dtype = torch.bfloat16

    torch_model = RmsNorm(dim=input_shape[-1], eps=1.0).to(dtype=dtype)
    torch.nn.init.normal_(torch_model.weight)

    parameters = TtRmsNormParameters.from_torch(torch_model.state_dict(), device=device)
    tt_model = TtRmsNorm(parameters, eps=torch_model.eps)

    torch_input_tensor = torch.randn(input_shape, dtype=dtype)

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.TILE_LAYOUT)

    torch_output = torch_model(torch_input_tensor)

    tt_output = tt_model(tt_input_tensor)
    tt_output_torch = ttnn.to_torch(tt_output)

    mse = torch.nn.functional.mse_loss(
        torch_output.to(dtype=torch.float32),
        tt_output_torch.to(dtype=torch.float32),
    ).item()
    logger.info(f"mse: {mse:.6f}")
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999_950)
