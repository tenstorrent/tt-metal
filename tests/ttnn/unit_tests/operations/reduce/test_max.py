# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = pytest.mark.use_module_device

import torch

import ttnn
from models.common.utility_functions import torch_random
from tests.ttnn.utils_for_testing import assert_allclose, assert_equal

TEST_PADDING_VALUE = -42


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64, 41, 10])
@pytest.mark.parametrize("w", [32, 64, 31, 18])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_max(device, batch_size, h, w, dim, dtype):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -100, 100, dtype=torch.bfloat16)
    torch_output_tensor, _ = torch.max(torch_input_tensor, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.max(input_tensor, dim=dim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)

    # test for equivalance
    assert_equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("batch_size1", [2])
@pytest.mark.parametrize("batch_size2", [32])
@pytest.mark.parametrize("h", [64, 15])
@pytest.mark.parametrize("w", [64, 22])
@pytest.mark.parametrize("dim", [-3])
def test_max_4d(device, batch_size1, batch_size2, h, w, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size1, batch_size2, h, w), -100, 100, dtype=torch.bfloat16)
    torch_output_tensor, _ = torch.max(torch_input_tensor, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.max(input_tensor, dim=dim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)

    # test for equivalance
    assert_equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [64, 15])
@pytest.mark.parametrize("w", [64, 22])
@pytest.mark.parametrize("dim", [-2, -1, 0, 1])
def test_max_2d(device, h, w, dim):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((h, w), -100, 100, dtype=torch.bfloat16)
    torch_output_tensor, _ = torch.max(torch_input_tensor, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.max(input_tensor, dim=dim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)

    # test for equivalance
    assert_equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64, 41, 37, 10])
@pytest.mark.parametrize("w", [32, 64, 31, 63, 18])
def test_max_global(device, batch_size, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -100, 100, dtype=torch.bfloat16)
    torch_output_tensor = torch.max(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.max(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    # test for equivalance
    assert_equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "input_shape_and_dim",
    [
        ((32, 32, 32, 64), -4),
        ((2, 32, 32, 64), -3),
        ((32, 32, 64), -3),
        ((1, 2, 3, 4), -1),
        ((2, 22, 37, 41), -4),
        ((2, 32, 64, 64), -3),
        ((2, 22, 37, 41), -3),
        ((2, 32, 64, 64), -2),
        ((2, 22, 37, 41), -1),
        ((2, 32, 64, 64), -1),
        ((2, 22, 37), -3),
        ((2, 22, 37), -2),
        ((2, 22, 37), -1),
        ((1, 6, 7), -3),
        ((32, 6, 7), -3),
    ],
)
@pytest.mark.parametrize("keepdim", [True, False])
def test_max_dim(device, input_shape_and_dim, keepdim):
    torch.manual_seed(0)
    input_shape, max_dim = input_shape_and_dim
    torch_input_tensor = torch_random(input_shape, -100, 100, dtype=torch.bfloat16)
    torch_output_tensor, _ = torch.max(torch_input_tensor, dim=max_dim, keepdim=keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.max(input_tensor, dim=max_dim, keepdim=keepdim)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)

    # test for equivalance
    assert_equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("input_shape", [(16, 2, 32, 3), (16, 2, 32, 24), (1, 1, 64, 64)])
@pytest.mark.parametrize("dim", [None, -1, -2])
@pytest.mark.parametrize("scalar", [1.0, 2.5, -2.5])
@pytest.mark.parametrize("fast_and_approximate_mode", [False, True], ids=["accurate", "fast"])
def test_max_fp32_fast_and_approximate_mode(device, input_shape, dim, scalar, fast_and_approximate_mode):
    """FLOAT32 max with both values of fast_and_approximate_mode.
    - False (default): accurate SFPU path - result matches torch exactly.
    - True: faster FPU/TF32 path - result is approximate.
    """
    torch.manual_seed(1)

    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
    torch_output_tensor = torch.amax(scalar * torch_input_tensor, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.max(input_tensor, fast_and_approximate_mode=fast_and_approximate_mode, dim=dim, scalar=scalar)
    output_tensor = ttnn.to_torch(ttnn.from_device(output_tensor)).reshape(torch_output_tensor.shape)

    # A negative scalar dispatches to min, and fp32 min is not routed to the SFPU yet,
    # so it stays on the FPU path.
    if fast_and_approximate_mode or scalar < 0 or device.arch() == ttnn.device.Arch.QUASAR:
        assert_allclose(torch_output_tensor, output_tensor, rtol=1e-3, atol=1e-2)
    else:
        assert_equal(torch_output_tensor, output_tensor)
