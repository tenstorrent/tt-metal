# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_ulp


def torch_polyval(input_tensor, coeff):
    curVal = 0
    for curValIndex in range(len(coeff) - 1):
        curVal = (curVal + coeff[curValIndex]) * input_tensor[0]
    return curVal + coeff[len(coeff) - 1]


@pytest.mark.parametrize("shape", [(1, 1, 32, 32)])
@pytest.mark.parametrize("coeff", [(1.5, 2.4, 6.7, 9.1)])
def test_polyval(device, shape, coeff):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)

    torch_output_tensor = torch_polyval(torch_input_tensor, coeff)

    input_tensor_a = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.polyval(input_tensor_a, coeff)
    output_tensor = ttnn.to_torch(output_tensor).squeeze(0)
    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=2)


@pytest.mark.parametrize("shape", [(1, 1, 32, 32)])
@pytest.mark.parametrize("coeff", [(3.0,), (3.0, 2.0)])
def test_polyval_honours_memory_config(device, shape, coeff):
    """polyval must return in the requested memory config, including the 1-coefficient path.

    Regression: `coeffs.size() == 1` returns early via `full_like` without the memory
    config, so the result inherited the input's. The multi-coefficient path was already
    correct, which is why it is parametrized here alongside as a control.

    The input is placed in L1 with DRAM requested; with matching configs the defect is
    invisible.
    """
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    output = ttnn.polyval(input_tensor, list(coeff), memory_config=ttnn.DRAM_MEMORY_CONFIG)

    assert (
        output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    ), f"{len(coeff)} coeff(s): requested {ttnn.DRAM_MEMORY_CONFIG} but landed in {output.memory_config()}"

    # with no memory_config both paths must fall back the same way, to the input's config
    default_output = ttnn.polyval(input_tensor, list(coeff))
    assert (
        default_output.memory_config() == ttnn.L1_MEMORY_CONFIG
    ), f"{len(coeff)} coeff(s): unset config should follow the input but landed in {default_output.memory_config()}"
