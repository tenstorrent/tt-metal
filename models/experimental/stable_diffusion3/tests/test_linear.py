# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

from ..tt.linear import TtLinear, TtLinearParameters


@pytest.mark.parametrize(
    ("batch_size", "input_dim", "output_dim"),
    [
        (32, 1536, 2048),
    ],
)
@pytest.mark.usefixtures("use_program_cache")
def test_linear(
    *,
    device: ttnn.Device,
    batch_size: int,
    input_dim: int,
    output_dim: int,
) -> None:
    dtype = torch.bfloat16

    torch_model = torch.nn.Linear(input_dim, output_dim).to(dtype=dtype)
    torch_model.eval()

    parameters = TtLinearParameters.from_torch(torch_model.state_dict(), device=device, dtype=ttnn.bfloat8_b)
    tt_model = TtLinear(parameters)

    torch_input_tensor = torch.randn((batch_size, input_dim), dtype=dtype)

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
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999_900)
