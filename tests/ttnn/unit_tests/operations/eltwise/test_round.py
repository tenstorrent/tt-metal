# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from functools import partial
from models.common.utility_functions import torch_random
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import to_tt_tensor
from tests.ttnn.utils_for_testing import flush_subnormal_values_to_zero, generate_all_bfloat16_bitpatterns

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "shape",
    (
        torch.Size([1, 1, 320, 320]),
        torch.Size([1, 1, 320, 384]),
        torch.Size([1, 3, 320, 384]),
    ),
)
@pytest.mark.parametrize("decimal", [-3, 6, -1, None])
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_round_new(shape, dtypes, decimal, device):
    torch.manual_seed(0)
    torch_dtype, tt_dtype = dtypes
    torch_input_tensor = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch_dtype), tt_dtype)(
        shape
    )

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=tt_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    torch_input_tensor = ttnn.to_torch(input_tensor)
    golden_function = ttnn.get_golden_function(ttnn.round)
    if decimal is None:
        torch_output_tensor = golden_function(torch_input_tensor)
        output_tensor = ttnn.round(input_tensor)
    else:
        torch_output_tensor = golden_function(torch_input_tensor, decimals=decimal)
        output_tensor = ttnn.round(input_tensor, decimals=decimal)
    output_tensor = ttnn.to_torch(output_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.999


@pytest.mark.parametrize("decimals", [-3, -1, 0, 1, 2, 3, 5, 7])
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
)
def test_round_all_bitpatterns(dtypes, decimals, device):
    """Every bfloat16 bit pattern through ttnn.round.

    The kernel scales by 10**decimals before rounding, which overflows for
    |x| > FLT_MAX / 10**decimals; the inverse scale cannot bring +/-inf back. Every such
    input is already an integer, so the result must be the input itself.
    """
    torch_dtype, tt_dtype = dtypes
    torch_input_tensor = flush_subnormal_values_to_zero(generate_all_bfloat16_bitpatterns(torch_dtype))
    finite = torch.isfinite(torch_input_tensor)

    input_tensor = to_tt_tensor(torch_input_tensor, device, dtype=tt_dtype)
    output_tensor = ttnn.to_torch(ttnn.round(input_tensor, decimals=decimals))

    # a finite input must never come back infinite
    assert torch.equal(torch.isfinite(output_tensor) & finite, finite)

    if decimals >= 0:
        # |x| >= 2**23 has no fractional part, so rounding to any number of decimal places
        # returns it unchanged
        integral = finite & (torch_input_tensor.abs() >= 2.0**23)
        assert torch.equal(output_tensor[integral], torch_input_tensor[integral])
