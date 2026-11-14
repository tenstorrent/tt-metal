# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Python binding coverage for ``ttml.ops.unary.sigmoid``."""

import numpy as np
import pytest
import torch

import ttnn
import ttml

pytestmark = pytest.mark.requires_device

SHAPE = (2, 1, 32, 32)


@pytest.fixture(autouse=True)
def _reset_graph():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


def _torch_sigmoid_and_grad(data):
    """Reference forward and d/dx, seeded with dL/dout = 1 to match ``backward()``."""
    x = torch.from_numpy(data).requires_grad_(True)
    out = torch.sigmoid(x)
    out.backward(torch.ones_like(out))
    return out.detach().numpy(), x.grad.numpy()


# Same tolerances as the C++ tests
RTOL = 8e-3
ATOL = 2e-2


@pytest.mark.parametrize("low, high", [(-1.0, 1.0), (-6.0, 6.0)], ids=["unit", "wide"])
def test_sigmoid_forward_backward(low, high):
    """The binding runs ttnn.sigmoid forward and records a node whose backward is ttnn.sigmoid_bw."""

    data = np.random.uniform(low, high, SHAPE).astype(np.float32)
    tensor = ttml.autograd.Tensor.from_numpy(data, layout=ttnn.Layout.TILE)
    tensor.set_requires_grad(True)

    result = ttml.ops.unary.sigmoid(tensor)
    expected, expected_grad = _torch_sigmoid_and_grad(data)
    np.testing.assert_allclose(result.to_numpy(ttnn.DataType.FLOAT32), expected, rtol=RTOL, atol=ATOL)

    result.backward(False)
    assert tensor.is_grad_initialized()
    grad = tensor.get_grad_tensor().to_numpy(ttnn.DataType.FLOAT32)
    np.testing.assert_allclose(grad, expected_grad, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
