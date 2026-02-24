# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_allclose


@pytest.mark.parametrize(
    "dtype",
    [
        "float32",
        "bfloat16",
    ],
)
@pytest.mark.parametrize("alpha_p, alpha_n", [(0.8, 0.8), (0.3, 0.1), (0.5, 1.0), (1.0, 0.5)])
def test_xielu(alpha_p, alpha_n, dtype, device):
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)
    torch_input = torch.randn([32, 32], dtype=torch_dtype)
    golden_fn = ttnn.get_golden_function(ttnn.xielu)
    torch_output = golden_fn(torch_input, alpha_p=alpha_p, alpha_n=alpha_n)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn.xielu(ttnn_input, alpha_p=alpha_p, alpha_n=alpha_n)
    ttnn_output = ttnn.to_torch(ttnn_output)

    if dtype == "float32":
        assert_allclose(torch_output, ttnn_output, rtol=6e-05, atol=1e-06)
    else:
        assert_with_ulp(torch_output, ttnn_output, 1)
