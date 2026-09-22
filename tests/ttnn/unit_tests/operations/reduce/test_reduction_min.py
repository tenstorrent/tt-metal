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


@pytest.mark.parametrize("input_shape, dim, keepdim", [((32, 32, 1, 2), -1, False), ((64, 512), -1, False)])
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
def test_min_fp32_accurate(device, input_shape, dim, scalar):
    """FLOAT32 min on the accurate SFPU path (the LLK MIN reduce) — result matches torch exactly.
    fast_and_approximate_mode=True is refused; see test_min_fp32_fast_mode_rejected.
    """
    torch.manual_seed(1)

    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
    torch_output_tensor = torch.amin(scalar * torch_input_tensor, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.min(input_tensor, dim=dim, scalar=scalar)
    output_tensor = ttnn.to_torch(ttnn.from_device(output_tensor)).reshape(torch_output_tensor.shape)

    if device.arch() == ttnn.device.Arch.QUASAR:
        assert_allclose(torch_output_tensor, output_tensor, rtol=1e-3, atol=1e-2)
    else:
        assert_equal(torch_output_tensor, output_tensor)


# The flag asks for the FPU, which has no min pool: fp32 min would lower to -max(-x), paying an
# extra negate pass and tf32 truncation and giving up the H-axis split. Measured never faster than
# the default across H and W, and up to 8.5x slower, so the flag has no right answer here.
@pytest.mark.parametrize("dim", [None, -1, -2])
def test_min_fp32_fast_mode_rejected(device, dim, expect_error):
    torch.manual_seed(1)
    input_tensor = ttnn.from_torch(
        torch.randn((1, 1, 64, 64), dtype=torch.float32), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32
    )
    with expect_error(RuntimeError, "does not support fast_and_approximate_mode=True on Float32"):
        ttnn.min(input_tensor, dim=dim, fast_and_approximate_mode=True)


# Without fp32_dest_acc_en there is no SFPU min to fall back to, so fp32 min stays on -max(-x):
# un-split and tf32-lossy, and accepted rather than refused.
@pytest.mark.parametrize("dim", [-1, -2])
def test_min_fp32_without_dest_acc_runs_unsplit(device, dim):
    torch.manual_seed(1)
    shape = (1, 1, 4096, 128)  # tall enough that the split would engage if it could
    torch_input_tensor = torch.randn(shape, dtype=torch.float32)
    torch_output_tensor = torch.amin(torch_input_tensor, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    output_tensor = ttnn.min(
        input_tensor,
        dim=dim,
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
        ),
    )
    output_tensor = ttnn.to_torch(ttnn.from_device(output_tensor)).reshape(torch_output_tensor.shape)
    # A 16-bit DEST packs the selected value through bfloat16, so it lands within one bf16 ulp.
    assert_allclose(torch_output_tensor, output_tensor, rtol=2**-7, atol=0.0)


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 1, 32, 256),  # Wt=8
        (1, 1, 256, 32),  # Ht=8
        (1, 1, 128, 544),  # Wt=17
        (1, 1, 60, 100),  # tile-unaligned
    ],
)
@pytest.mark.parametrize("dim", [-1, -2, None])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["dest16", "dest32"])
def test_min_bfloat16_dest_modes(device, input_shape, dim, fp32_dest_acc_en):
    """bfloat16 min over both DEST widths. The SFPU reduce sizes its chunk from DEST capacity,
    so fp32_dest_acc_en halves it; the shapes straddle both chunk boundaries."""
    torch.manual_seed(0)

    torch_input_tensor = torch_random(input_shape, -100, 100, dtype=torch.bfloat16)
    torch_output_tensor = torch.amin(torch_input_tensor, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    # Padding sits below every input value, so a chunk reading past the valid region returns it.
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    output_tensor = ttnn.min(
        input_tensor,
        dim=dim,
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(),
            # HiFi4 with fp32_dest_acc_en can return wrong results on Wormhole.
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
        ),
    )
    output_tensor = ttnn.to_torch(ttnn.from_device(output_tensor)).reshape(torch_output_tensor.shape)

    assert_equal(torch_output_tensor, output_tensor)
