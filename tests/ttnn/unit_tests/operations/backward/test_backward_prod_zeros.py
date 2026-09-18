# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 1, 1, 64)])
def test_prod_bw_with_zeros(device, shape):
    """Verify issue #54551: ttnn.prod_bw returns finite gradients for inputs containing zeros."""
    torch.manual_seed(0)
    # Create tensor containing multiple exact zeros
    pyt_input = torch.randn(shape, dtype=torch.bfloat16)
    pyt_input[..., 0] = 0.0
    pyt_input[..., 5] = 0.0
    pyt_input.requires_grad = True

    pyt_grad = torch.randn(shape, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(pyt_input, device=device, layout=ttnn.TILE_LAYOUT)
    tt_grad = ttnn.from_torch(pyt_grad, device=device, layout=ttnn.TILE_LAYOUT)

    grad_res = ttnn.prod_bw(tt_grad, tt_input)
    assert len(grad_res) >= 1
    out_torch = ttnn.to_torch(grad_res[0])

    # Assert no non-finite (NaN or Inf) values in backward pass
    assert torch.all(torch.isfinite(out_torch)), "prod_bw gradient produced non-finite values!"
