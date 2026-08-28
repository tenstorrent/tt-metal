# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# Both operations should properly handle IEEE-754 semantics:
# max(inf, inf) + ... = inf
# max(-inf, -inf) + ... = -inf
# max(inf, -inf) + ... = inf
# max(NaN, ...) + ... = NaN

@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("op", [ttnn.logaddexp, ttnn.logaddexp2])
def test_eltwise_binary_infinities_and_edges(device, dtype, op):
    # Determine the torch equivalent op
    torch_op = torch.logaddexp if op == ttnn.logaddexp else torch.logaddexp2

    # Map ttnn dtype to torch dtype
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16

    a_values = [
        float("inf"), float("-inf"), float("inf"), float("-inf"),
        float("inf"), float("-inf"), 10.0, 10.0,
        float("nan"), 10.0, float("nan"), float("nan"), float("inf"),
        0.0, -0.0, 0.0, -0.0,
        100.0, 0.0, 100.0, -100.0, -100.0,
        1e4, -1e4, 1000.0, 5.0,
    ]
    b_values = [
        float("inf"), float("-inf"), float("-inf"), float("inf"),
        10.0, 10.0, float("inf"), float("-inf"),
        10.0, float("nan"), float("nan"), float("inf"), float("nan"),
        -0.0, 0.0, 0.0, -0.0,
        0.0, 100.0, 100.0, -100.0, 0.0,
        1e4, -1e4, 999.0, 3.0,
    ]
    a_tensor = torch.tensor([a_values], dtype=torch_dtype)
    b_tensor = torch.tensor([b_values], dtype=torch_dtype)

    # Calculate golden result with torch
    expected = torch_op(a_tensor, b_tensor)

    # Calculate ttnn result
    input_tensor_a = ttnn.from_torch(a_tensor, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    input_tensor_b = ttnn.from_torch(b_tensor, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

    output_tensor = op(input_tensor_a, input_tensor_b)
    actual = ttnn.to_torch(output_tensor)

    # Extract raw lists for explicit per-element comparison of NaNs and Infs
    expected_list = expected.flatten().tolist()
    actual_list = actual.flatten().tolist()

    for i in range(len(expected_list)):
        e = expected_list[i]
        a = actual_list[i]

        if math.isnan(e):
            assert math.isnan(a), f"Index {i} (a={a_values[i]}, b={b_values[i]}): Expected NaN, got {a}"
        elif math.isinf(e):
            assert math.isinf(a) and (e == a), f"Index {i} (a={a_values[i]}, b={b_values[i]}): Expected {e}, got {a}"
        else:
            # Check numerical accuracy for finite values
            assert math.isfinite(a), f"Index {i} (a={a_values[i]}, b={b_values[i]}): Expected finite {e}, got {a}"

    # Overall PCC check on finite values
    finite_mask = torch.isfinite(expected)
    if finite_mask.any():
        assert_with_pcc(expected[finite_mask], actual[finite_mask], pcc=0.999)
