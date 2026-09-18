# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = pytest.mark.use_module_device

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_allclose, assert_equal, assert_numeric_metrics
from models.common.utility_functions import torch_random

TEST_PADDING_VALUE = -142


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64, 41, 37])
@pytest.mark.parametrize("w", [32, 64, 31, 63])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_min(device, batch_size, h, w, dim, keepdim, dtype):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -100, 100, dtype=torch.bfloat16)
    torch_output_tensor, _ = torch.min(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.min(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    # test for equivalance
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
        check_ulp=True,
    )


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64, 41, 37])
@pytest.mark.parametrize("w", [32, 64, 31, 63])
def test_min_global(device, batch_size, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -100, 100, dtype=torch.bfloat16)
    torch_output_tensor = torch.min(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.min(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)

    # test for equivalance
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
        check_ulp=True,
    )


@pytest.mark.parametrize("input_shape, dim, keepdim", [((512, 1024, 1, 2), -1, False), ((64, 512), -1, False)])
def test_min_row_major(device, input_shape, dim, keepdim):
    """Test ttnn.min with ROW_MAJOR layout (issue #32829: +inf padding during tilization)."""
    torch.manual_seed(0)
    torch_input_tensor = torch_random(input_shape, -100, 100, dtype=torch.bfloat16)
    torch_output_tensor, _ = torch.min(torch_input_tensor, dim=dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
    output_tensor = ttnn.min(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_torch(output_tensor)

    # test for equivalance
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
        check_ulp=True,
    )


@pytest.mark.parametrize(
    "input_shape",
    [
        (32, 32, 32, 32, 32),
        (3, 6, 40, 64, 32),
        (3, 6, 40, 63, 20),
    ],
)
def test_min_multi_dim(device, input_shape):
    """Test from issue #40854: ttnn.min produces incorrect results for certain tensor shapes and dimensions."""
    dims = (-2, -1)
    torch.manual_seed(0)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.amin(torch_input_tensor, dim=dims, keepdim=True)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    output_tensor = ttnn.min(input_tensor, dim=dims, keepdim=True)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)

    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-09,
        check_ulp=True,
    )


@pytest.mark.parametrize("input_shape", [(32, 32), (16, 2, 32, 3), (16, 2, 32, 24), (1, 1, 64, 64)])
@pytest.mark.parametrize("dim", [None, -1, -2])
@pytest.mark.parametrize("scalar", [1.0, 2.5, -2.5])
@pytest.mark.parametrize("fast_and_approximate_mode", [False, True], ids=["accurate", "fast"])
def test_min_fp32_fast_and_approximate_mode(device, input_shape, dim, scalar, fast_and_approximate_mode):
    """FLOAT32 min with both values of fast_and_approximate_mode.
    - False (default): accurate SFPU path (LLK MIN reduce) - result matches torch exactly.
    - True: faster FPU/TF32 path via -MAX(-x) - result is approximate.
    """
    torch.manual_seed(1)

    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
    torch_output_tensor = torch.amin(scalar * torch_input_tensor, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.min(input_tensor, fast_and_approximate_mode=fast_and_approximate_mode, dim=dim, scalar=scalar)
    output_tensor = ttnn.to_torch(ttnn.from_device(output_tensor)).reshape(torch_output_tensor.shape)

    if fast_and_approximate_mode or device.arch() == ttnn.device.Arch.QUASAR:
        assert_allclose(torch_output_tensor, output_tensor, rtol=1e-3, atol=1e-2)
    else:
        assert_equal(torch_output_tensor, output_tensor)
