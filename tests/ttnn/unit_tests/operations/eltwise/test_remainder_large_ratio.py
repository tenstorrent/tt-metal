# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "dividend, divisor",
    [
        (1.42606e7, 3.0),
        (5.70425e7, 3.0),
        (-5.70425e7, 3.0),
        (5.70425e7, -3.0),
        (-5.70425e7, -3.0),
    ],
)
def test_remainder_large_ratio(device, dividend, divisor):
    torch_input = torch.tensor([[dividend]], dtype=torch.float32)
    expected = torch.remainder(torch_input, divisor)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.remainder(input_tensor, divisor))

    assert torch.equal(actual, expected), (
        f"remainder({dividend}, {divisor}) produced {actual.item()}, expected {expected.item()}"
    )


def test_remainder_large_ratio_boundary_sweep(device):
    divisor = 3.0
    quotients = torch.tensor(
        [2**22 - 1, 2**22, 2**23 - 1, 2**23, 2**23 + 1, 2**24, 2**25, 2**26],
        dtype=torch.float32,
    )
    exact_multiples = quotients * divisor
    torch_input = torch.nextafter(exact_multiples, torch.full_like(exact_multiples, float("inf"))).reshape(1, -1)
    expected = torch.remainder(torch_input, divisor)

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.remainder(input_tensor, divisor))

    assert torch.equal(actual, expected)
