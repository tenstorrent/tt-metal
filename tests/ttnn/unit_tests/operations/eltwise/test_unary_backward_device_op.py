# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the shared unary-backward device operation
(ttnn/cpp/ttnn/operations/eltwise/unary_backward/device).

The device operation is written once for every gradient in UnaryBackwardOpType; sigmoid_bw is
the first op routed through it. These tests cover what the shared layer owns -- output specs,
dtype/layout validation, program-cache keying and the fused kernel's accuracy -- so that
porting the next gradient only needs an accuracy test of its own.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def _torch_sigmoid_bw(grad, inp):
    x = inp.to(torch.float32).detach().clone().requires_grad_(True)
    torch.sigmoid(x).backward(grad.to(torch.float32))
    return x.grad


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),  # single tile
        (1, 1, 320, 384),
        (1, 3, 320, 384),
        (2, 2, 64, 96),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
def test_sigmoid_bw_matches_torch(shape, dtype, device):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32) * 4.0
    torch_grad = torch.randn(shape, dtype=torch.float32) * 10.0

    input_tensor = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(torch_grad, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.sigmoid_bw(grad_tensor, input_tensor)[0]

    assert output.dtype == input_tensor.dtype
    assert tuple(output.shape) == shape

    # bfloat8_b carries a shared exponent per 16-element block, so it needs a looser bar than
    # the bit-addressable float types.
    expected_pcc = 0.99 if dtype == ttnn.bfloat8_b else 0.9995
    assert_with_pcc(_torch_sigmoid_bw(torch_grad, torch_input), ttnn.to_torch(output), expected_pcc)


@pytest.mark.parametrize(
    "dtype, rtol",
    [(ttnn.bfloat16, 2e-2), (ttnn.float32, 1e-6), (ttnn.bfloat8_b, 6e-2)],
)
def test_sigmoid_bw_sub_tile_shape(dtype, rtol, device):
    """A (1, 1, 1, 1) logical shape padded up to one tile. Asserted on the value rather than by
    PCC: a one-element tensor has zero variance, so comp_pcc falls back to an allclose at
    float32 tolerances and reports PCC 0.0 for a result that is correct to within the dtype's
    own quantisation of the input."""
    torch.manual_seed(0)
    torch_input = torch.randn((1, 1, 1, 1), dtype=torch.float32) * 4.0
    torch_grad = torch.randn((1, 1, 1, 1), dtype=torch.float32) * 10.0

    input_tensor = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(torch_grad, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.to_torch(ttnn.sigmoid_bw(grad_tensor, input_tensor)[0])

    # Reference from the operands as the device actually holds them, so the bar measures the
    # kernel and not the operand rounding that happened in from_torch.
    quantised_input = ttnn.to_torch(input_tensor).float()
    quantised_grad = ttnn.to_torch(grad_tensor).float()
    torch.testing.assert_close(output.float(), _torch_sigmoid_bw(quantised_grad, quantised_input), rtol=rtol, atol=1e-7)


def test_sigmoid_bw_bfloat16_keeps_intermediates_in_float32(device):
    """s(1 - s) cancels: for x beyond about +/-6 a bfloat16 s is within an ulp of 1.0 and
    (1 - s) keeps only a couple of significant bits. The fused kernel holds s in float32 DEST
    (UnaryBackwardKernelSpec::force_fp32_dest_acc), which the composite could not do because it
    had to write each intermediate back to L1 as bfloat16. This pins that gain."""
    torch.manual_seed(0)
    shape = (1, 1, 32, 32)
    torch_input = torch.randn(shape, dtype=torch.float32) * 4.0
    torch_grad = torch.randn(shape, dtype=torch.float32) * 10.0

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.to_torch(ttnn.sigmoid_bw(grad_tensor, input_tensor)[0]).float()
    expected = _torch_sigmoid_bw(ttnn.to_torch(grad_tensor).float(), ttnn.to_torch(input_tensor).float())

    relative_error = ((output - expected).abs() / expected.abs().clamp(min=1e-30)).max()
    # Composing the gradient out of bfloat16 forward ops reaches a relative error above 1.0 on
    # this input; keeping the intermediates in float32 holds it near the bfloat16 epsilon.
    assert relative_error < 0.05, f"bfloat16 sigmoid_bw max relative error regressed to {relative_error}"


@pytest.mark.parametrize(
    "grad_dtype, input_dtype",
    [
        (ttnn.bfloat16, ttnn.float32),
        (ttnn.float32, ttnn.bfloat16),
        (ttnn.bfloat8_b, ttnn.bfloat16),
    ],
)
def test_sigmoid_bw_mixed_operand_dtypes(grad_dtype, input_dtype, device):
    """grad_output and input need not share a dtype -- the composite accepted that, so the fused
    kernel has to as well. It only can if the unpacker reconfigures between the two buffers;
    with reconfig disabled these combinations return nan or garbage."""
    torch.manual_seed(0)
    shape = (1, 1, 32, 32)
    torch_input = torch.randn(shape, dtype=torch.float32) * 2.0
    torch_grad = torch.randn(shape, dtype=torch.float32) * 3.0

    input_tensor = ttnn.from_torch(torch_input, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(torch_grad, dtype=grad_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.to_torch(ttnn.sigmoid_bw(grad_tensor, input_tensor)[0]).float()

    assert output.isfinite().all(), f"mixed {grad_dtype}/{input_dtype} produced non-finite values"

    # Compared against the operands as quantised, and by PCC rather than elementwise: bfloat8_b
    # shares one exponent per 16 values, so an individually small gradient legitimately flushes
    # to zero (the composite flushes the same elements).
    expected = _torch_sigmoid_bw(ttnn.to_torch(grad_tensor).float(), ttnn.to_torch(input_tensor).float())
    assert_with_pcc(expected, output, 0.999)


@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_sigmoid_bw_honours_memory_config(memory_config, device):
    torch.manual_seed(0)
    shape = (1, 1, 64, 64)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_grad = torch.randn(shape, dtype=torch.float32)

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(torch_grad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.sigmoid_bw(grad_tensor, input_tensor, memory_config=memory_config)[0]

    assert output.memory_config().buffer_type == memory_config.buffer_type
    assert_with_pcc(_torch_sigmoid_bw(torch_grad, torch_input), ttnn.to_torch(output), 0.9995)


def test_sigmoid_bw_saturates_like_torch(device):
    """The derivative s(1-s) underflows to zero well before sigmoid itself saturates, and the
    fused kernel must reach zero at the same inputs torch does rather than emitting inf/nan."""
    values = [-100.0, -40.0, -20.0, -1.0, 0.0, 1.0, 20.0, 40.0, 100.0]
    torch_input = torch.tensor(values, dtype=torch.float32).repeat(32, 4)[:32, :32].reshape(1, 1, 32, 32)
    torch_grad = torch.full_like(torch_input, 3.0)

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(torch_grad, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.to_torch(ttnn.sigmoid_bw(grad_tensor, input_tensor)[0])
    expected = _torch_sigmoid_bw(torch_grad, torch_input)

    assert torch.isfinite(output).all(), "fused sigmoid_bw produced a non-finite gradient"
    torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-6)


def test_sigmoid_bw_program_cache_distinguishes_dtypes(device):
    """One device operation now serves several gradients and dtypes, so the program hash has to
    separate them. Two dtypes back to back must add two entries and both stay correct."""
    torch.manual_seed(0)
    shape = (1, 1, 96, 128)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_grad = torch.randn(shape, dtype=torch.float32)
    expected = _torch_sigmoid_bw(torch_grad, torch_input)

    start = device.num_program_cache_entries()
    for dtype, pcc in [(ttnn.bfloat16, 0.9995), (ttnn.float32, 0.9995), (ttnn.bfloat16, 0.9995)]:
        input_tensor = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        grad_tensor = ttnn.from_torch(torch_grad, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        output = ttnn.sigmoid_bw(grad_tensor, input_tensor)[0]
        assert_with_pcc(expected, ttnn.to_torch(output), pcc)

    # bfloat16 then float32 then bfloat16 again: two distinct programs, the third a cache hit.
    assert device.num_program_cache_entries() - start == 2


def test_sigmoid_bw_rejects_row_major(device, expect_error):
    """The shared validation guards the cache-key holes the factory has: layout reaches the
    kernels as a compile-time page size, so a ROW_MAJOR operand must be rejected outright."""
    shape = (1, 1, 32, 32)
    torch_input = torch.randn(shape, dtype=torch.float32)

    tile = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    row_major = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with expect_error(RuntimeError, "TILE layout"):
        ttnn.sigmoid_bw(row_major, tile)


def test_sigmoid_bw_rejects_mismatched_shapes(device, expect_error):
    """The reader derives its tile count from the input alone, so a smaller grad_output would be
    read past the end of its allocation."""
    tile_a = ttnn.from_torch(
        torch.randn((1, 1, 64, 64), dtype=torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tile_b = ttnn.from_torch(
        torch.randn((1, 1, 32, 32), dtype=torch.float32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    with expect_error(RuntimeError, "same logical shape"):
        ttnn.sigmoid_bw(tile_b, tile_a)
