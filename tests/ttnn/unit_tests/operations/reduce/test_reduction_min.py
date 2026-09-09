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


@pytest.mark.parametrize("input_shape", [(32, 32), (16, 2, 32, 3), (16, 2, 32, 24), (1, 1, 64, 64)])
@pytest.mark.parametrize("dim", [None, -1, -2])
@pytest.mark.parametrize("scalar", [1.0, 2.5, -2.5])
@pytest.mark.parametrize("fast_and_approximate_mode", [False, True], ids=["accurate", "fast"])
def test_min_bf16_fast_and_approximate_mode(device, input_shape, dim, scalar, fast_and_approximate_mode):
    """BFLOAT16 min with both values of fast_and_approximate_mode.

    Unlike FLOAT32 the switch is not about precision: the FPU has no min pool, so
    - False (default): SFPU LLK MIN reduce.
    - True: FPU -MAX(-x) via the fused-negate kernels.
    Both select an input element, so an unscaled min is exact either way. A non-unit scalar is a
    post-multiply packed back to bf16, which lands within a ULP of scaling before the reduce.
    """
    torch.manual_seed(1)

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.amin(scalar * torch_input_tensor.float(), dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.min(input_tensor, fast_and_approximate_mode=fast_and_approximate_mode, dim=dim, scalar=scalar)
    output_tensor = ttnn.to_torch(ttnn.from_device(output_tensor)).float().reshape(torch_output_tensor.shape)

    if scalar == 1.0:
        assert_equal(torch_output_tensor, output_tensor)
    else:
        assert_allclose(torch_output_tensor, output_tensor, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("scalar", [1.0, 2.5])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1024, 160),  # Ht=32, the split threshold
        (1, 1, 1050, 160),  # non-aligned H — past-the-end slices carry the +inf identity
        (1, 1, 9216, 145),  # non-aligned W, deep H
        (2, 3, 1024, 40),  # NC=6
        (1, 1, 9216, 1024),  # Wt=32, only two slices
    ],
)
def test_min_bf16_sfpu_matches_fpu_h_axis_split(device, shape, scalar):
    """Tall-H bf16 min: only the SFPU path takes the H-axis split (fast mode sets negate, which the
    split declines), so the two engines running the same shape is what proves the split correct."""
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_input = ttnn.fill_implicit_tile_padding(tt_input, TEST_PADDING_VALUE)

    kw = {"dim": -2, "keepdim": False, "scalar": scalar}
    sfpu = ttnn.to_torch(ttnn.min(tt_input, fast_and_approximate_mode=False, **kw)).float()
    fpu = ttnn.to_torch(ttnn.min(tt_input, fast_and_approximate_mode=True, **kw)).float()
    assert torch.equal(sfpu, fpu), f"engine mismatch: {(sfpu - fpu).abs().max()}"

    if scalar == 1.0:
        reference = torch.amin(ttnn.to_torch(tt_input).float()[:, :, : shape[2], :], dim=-2)
        assert torch.equal(reference, sfpu), f"SFPU min mismatch: {(reference - sfpu).abs().max()}"
