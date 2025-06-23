# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import random
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    compare_pcc,
)


@pytest.mark.parametrize("ttnn_function", [ttnn.add, ttnn.sub, ttnn.multiply, ttnn.div])
@pytest.mark.parametrize("testing_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("input_a", [1, -1, 0, float("inf"), -float("inf"), float("nan")])
@pytest.mark.parametrize("input_b", [float("inf"), -float("inf"), float("nan"), 1, -1, 0])
def test_nan_inf_add(input_a, input_b, testing_dtype, ttnn_function, device):
    torch_dtype = getattr(torch, testing_dtype)
    ttnn_dtype = getattr(ttnn, testing_dtype)
    in_data1 = torch.tensor([input_a], dtype=torch_dtype)
    in_data2 = torch.tensor([input_b], dtype=torch_dtype)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn_function)
    golden_tensor = golden_function(in_data1, in_data2)
    print("testing_dtype    : ", testing_dtype)
    print("input_a          : ", input_a)
    print("input_b          : ", input_b)
    print("TT Result        : ", output_tensor)
    print("PyTorch Result   : ", golden_tensor)
    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass
