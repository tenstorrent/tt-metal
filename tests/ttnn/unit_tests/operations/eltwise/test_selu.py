import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import (
    assert_with_ulp,
    assert_allclose,
    generate_all_bfloat16_bitpatterns,
    flush_subnormal_values_to_zero,
)


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, 32, 32],
        [1, 1, 320, 384],
        [1, 3, 320, 384],
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_selu(device, input_shape, dtype):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    # Use a mix of positive and negative values in a practical range
    torch_input = torch.randn(input_shape, dtype=torch_dtype) * 3.0

    torch_output = torch.nn.functional.selu(torch_input.float())
    expected = torch_output.to(torch_dtype)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.selu(tt_input)
    actual = ttnn.to_torch(tt_output).to(torch_dtype)

    assert_allclose(expected, actual, rtol=1.6e-2, atol=1e-2)
