# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp
import math
import pytest


@pytest.mark.parametrize("val_a, val_b", [(1.0, 0.0), (-1.0, 0.0), (0.0, 0.0)])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("approx", [True, False])
def test_div_zero(device, val_a, val_b, dtype, approx):
    torch_dtype = getattr(torch, dtype)
    tt_dtype = getattr(ttnn, dtype)

    if val_a == 0.0 and val_b == 0.0 and dtype == "bfloat16":
        pytest.skip("Skipping test for 0/0 on bfloat16")

    x_torch = torch.tensor([[val_a]], dtype=torch_dtype)
    y_torch = torch.tensor([[val_b]], dtype=torch_dtype)

    golden_fn = ttnn.get_golden_function(ttnn.divide)
    z_torch = golden_fn(x_torch, y_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    z_tt_div = ttnn.divide(x_tt, y_tt, use_legacy=None, fast_and_approximate_mode=approx)
    tt_out = ttnn.to_torch(z_tt_div)

    # Note: torch.equal return false for if both tensors are nan
    # This is why we use assert_with_ulp to test for equality

    if not approx:
        assert_with_ulp(z_torch, tt_out, 0, allow_nonfinite=True)
