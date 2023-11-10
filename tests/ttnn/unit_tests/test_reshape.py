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
    tt_output = ttnn.reshape(activations, (1, 1, h, w))
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
