# SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_allclose

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, 32, 32],
        [1, 1, 320, 384],
        [1, 3, 320, 384],
    ],
)
@pytest.mark.parametrize("lambd", [0.5, 1.0, 0.0])
def test_softshrink_bfloat16(device, input_shape, lambd):
    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16) * 3.0

    torch_output = torch.nn.functional.softshrink(torch_input, lambd=lambd)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.softshrink(tt_input, lambd=lambd)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, pcc=0.99)
    assert_allclose(torch_output, tt_output, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, 32, 32],
        [1, 3, 320, 384],
    ],
)
def test_softshrink_default_lambd(device, input_shape):
    torch.manual_seed(42)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16) * 3.0

    torch_output = torch.nn.functional.softshrink(torch_input, lambd=0.5)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.softshrink(tt_input)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, pcc=0.99)
    assert_allclose(torch_output, tt_output, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, 32, 32],
        [1, 1, 320, 384],
    ],
)
@pytest.mark.parametrize("lambd", [0.5, 1.0])
def test_softshrink_memory_config(device, input_shape, lambd):
    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16) * 3.0

    torch_output = torch.nn.functional.softshrink(torch_input, lambd=lambd)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.softshrink(tt_input, lambd=lambd, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, pcc=0.99)
    assert_allclose(torch_output, tt_output, rtol=1.6e-2, atol=1e-2)
