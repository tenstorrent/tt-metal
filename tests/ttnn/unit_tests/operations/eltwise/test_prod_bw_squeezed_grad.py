# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# ttnn.prod_bw must accept the gradient shape torch produces for torch.prod(x, dim, keepdim=False),
# i.e. the reduced dim removed, as well as the keepdim shape.

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("input_shape", [(2, 3, 32, 32), (1, 4, 64, 32)])
@pytest.mark.parametrize("dim", [-4, -3, -2, -1, 0, 1, 2, 3])
@pytest.mark.parametrize("grad_form", ["squeezed", "keepdim"])
def test_prod_bw_squeezed_grad(input_shape, dim, grad_form, device):
    torch.manual_seed(0)
    in_data = (torch.randn(input_shape) + 2.0).bfloat16().requires_grad_(True)
    keepdim = grad_form == "keepdim"
    out = torch.prod(in_data, dim=dim, keepdim=keepdim)
    grad_data = torch.randn(out.shape).bfloat16()
    out.backward(gradient=grad_data)

    input_tensor = ttnn.from_torch(in_data.detach(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(grad_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    result = ttnn.prod_bw(grad_tensor, input_tensor, dim=dim)
    result = ttnn.to_torch(result[0])
    assert_with_pcc(in_data.grad, result, 0.99)
