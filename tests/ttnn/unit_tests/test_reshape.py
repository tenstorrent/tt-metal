# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_reshape(device, h, w):
    torch_activations = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_output = torch_activations.reshape(1, 1, w, h)

    activations = ttnn.from_torch(torch_activations)
    tt_output = ttnn.reshape(activations, (1, 1, w, h))
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.9999)
    # assert torch.allclose(torch_output, tt_output, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_reshape_negative_1(device, h, w):
    torch_activations = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    torch_output = torch_activations.reshape(-1)
    # activations.reshape(-1) is currently not supported

    activations = ttnn.from_torch(torch_activations)
    tt_output = ttnn.reshape(activations, (h * w,))  # TODO: allow passing in -1
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.9999)
    # assert torch.allclose(torch_output, tt_output, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("n", [32, 32])
@pytest.mark.parametrize("c", [2 * 32, 2 * 32])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("w", [1, 4])
def test_reshape_in_4D(device, n, c, h, w):
    torch_activations = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output = torch_activations.reshape(h, w, n, c)

    activations = ttnn.from_torch(torch_activations)
    tt_output = ttnn.reshape(activations, (h, w, n, c))
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.skip(reason="Issue #4007")
def test_permute_reshape(device):
    input_shape = (1, 4, 64, 32)
    output_shape = (1, 64, 128)

    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_output = torch.permute(torch_input, (0, 2, 1, 3))
    torch_output = torch.reshape(torch_output, output_shape)

    tt_input = ttnn.from_torch(torch_input)
    tt_input = ttnn.to_device(tt_input, device)
    tt_output = ttnn.permute(tt_input, (0, 2, 1, 3))
    tt_output = ttnn.reshape(tt_input, output_shape)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.9999)
