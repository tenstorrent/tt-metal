# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, construct_pcc_assert_message
from models.utility_functions import torch_random, comp_allclose


@pytest.mark.parametrize("batch_size", [1, 16, 1, 16])
@pytest.mark.parametrize("h", [32, 64, 41, 37])
@pytest.mark.parametrize("w", [32, 64, 31, 63])
@pytest.mark.parametrize("dim", [-1, -2])
def test_mean(device, batch_size, h, w, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, keepdim=True, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.mean(input_tensor, dim=dim)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("batch_size", [(1,), (4,), (64, 4), None])
@pytest.mark.parametrize("h", [1, 32, 64, 41, 37])
@pytest.mark.parametrize("w", [32, 64, 31, 63])
def test_mean_without_dim(device, batch_size, h, w):
    torch.manual_seed(0)
    input_shape = (*batch_size, h, w) if batch_size else (h, w)

    torch_input_tensor = torch_random(input_shape, -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, None, True)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.mean(input_tensor, keepdim=True)
    output_tensor = ttnn.to_torch(output_tensor)
    # PCC does not work for a single value. Assert on allclose.
    # visit issue: https://github.com/tenstorrent/tt-metal/issues/16454 for why tolerance values are changed
    close_passed, close_message = comp_allclose(torch_output_tensor, output_tensor, rtol=0.001, atol=0.00139)
    if not close_passed:
        print(f"Found mismatch: torch_output_tensor {torch_output_tensor}\n output_tensor {output_tensor}")
    assert close_passed, construct_pcc_assert_message(close_message, torch_output_tensor, output_tensor)
