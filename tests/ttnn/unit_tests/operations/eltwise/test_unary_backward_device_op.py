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
    """Reference gradient, computed the same way ttnn's registered golden function does.

    The golden function for sigmoid_bw is autograd-based, so routing sigmoid_bw through the
    shared device operation does not change it and it needs no update. This local helper exists
    only so a test can evaluate the reference at float32 against operands it chooses (e.g. the
    operands as the device quantised them), which the registered golden -- taking the caller's
    tensors as-is -- cannot do.

    Note both agree on the shape contract: autograd requires the gradient to match the input
    exactly, which is why the device operation TT_FATALs on unequal shapes rather than
    broadcasting.
    """
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
# Accuracy bar per dtype. The bit-addressable floats are checked ELEMENTWISE: PCC is a
# correlation over the whole tile, so it can mask a localized per-element error, and these
# dtypes can resolve one. The block float types keep PCC -- they carry one shared exponent per
# 16 values, so an individually small element legitimately flushes to zero, and an elementwise
# bound there would be measuring the storage format rather than the kernel.
#
# The elementwise bars are set from measurement with headroom, not from a dtype epsilon:
#   bfloat16  max relative error 3.89e-03  (~2^-8, i.e. rounding the result to bfloat16)
#   float32   max relative error 3.45e-04  (the SFPU sigmoid's own accuracy, well above
#                                           float32 eps -- a true ULP bound would not hold)
# both steady across every shape in the matrix.
@pytest.mark.parametrize(
    "dtype, rtol, expected_pcc",
    [
        (ttnn.bfloat16, 8e-3, None),
        (ttnn.float32, 1e-3, None),
        (ttnn.bfloat8_b, None, 0.99),
        (ttnn.bfloat4_b, None, 0.93),
    ],
)
def test_sigmoid_bw_matches_torch(shape, dtype, rtol, expected_pcc, device):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32) * 4.0
    torch_grad = torch.randn(shape, dtype=torch.float32) * 10.0

    input_tensor = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(torch_grad, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.sigmoid_bw(grad_tensor, input_tensor)[0]

    assert output.dtype == input_tensor.dtype, f"output dtype {output.dtype} != input dtype {input_tensor.dtype}"
    assert tuple(output.shape) == shape, f"output shape {tuple(output.shape)} != requested {shape}"

    if expected_pcc is not None:
        assert_with_pcc(_torch_sigmoid_bw(torch_grad, torch_input), ttnn.to_torch(output), expected_pcc)
        return

    # Referenced against the operands as the device holds them, so the bar measures the kernel
    # and not the rounding from_torch already applied to the inputs.
    expected = _torch_sigmoid_bw(ttnn.to_torch(grad_tensor).float(), ttnn.to_torch(input_tensor).float())
    torch.testing.assert_close(ttnn.to_torch(output).float(), expected, rtol=rtol, atol=1e-4)


@pytest.mark.parametrize(
    "dtype, rtol",
    # Same bars as the dtype matrix above, for the same reason: what is measured is the SFPU
    # sigmoid's own accuracy, which is architecture-dependent and well above a dtype epsilon.
    # float32 was originally pinned at 1e-6 from a single Wormhole element that happened to come
    # out exact; Blackhole resolves that element to 2.83e-05 relative (1.74e-07 absolute), so the
    # bar has to reflect the kernel rather than one measurement on one architecture.
    [(ttnn.bfloat16, 2e-2), (ttnn.float32, 1e-3), (ttnn.bfloat8_b, 6e-2)],
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
    torch.testing.assert_close(output.float(), _torch_sigmoid_bw(quantised_grad, quantised_input), rtol=rtol, atol=1e-6)


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

    assert (
        output.memory_config().buffer_type == memory_config.buffer_type
    ), f"output landed in {output.memory_config().buffer_type}, requested {memory_config.buffer_type}"
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
    added = device.num_program_cache_entries() - start
    assert added == 2, (
        f"expected 2 program cache entries (one per dtype, third call a hit), got {added}; "
        "more means the hash separates runs it should share, fewer means it collides dtypes"
    )


@pytest.mark.parametrize("shard_inputs", [True, False], ids=["sharded_inputs", "sharded_output_request"])
def test_sigmoid_bw_sharded_keeps_working(shard_inputs, device):
    """Sharded calls must keep working. The fused device operation is interleaved-only, so the
    composite layer routes sharded operands -- or a request for a sharded output -- to the op
    composition that supports them. Without that fallback, a previously valid sharded call
    raises, since the shared validation rejects sharded tensors."""
    torch.manual_seed(0)
    shape = (1, 1, 256, 32)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_grad = torch.randn(shape, dtype=torch.float32)

    shard_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 7))}),
            [32, 32],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    operand_memory_config = shard_memory_config if shard_inputs else ttnn.DRAM_MEMORY_CONFIG

    input_tensor = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=operand_memory_config
    )
    grad_tensor = ttnn.from_torch(
        torch_grad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=operand_memory_config
    )

    output = ttnn.sigmoid_bw(grad_tensor, input_tensor, memory_config=shard_memory_config)[0]

    assert (
        output.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    ), f"expected a height-sharded result, got {output.memory_config().memory_layout}"
    # Looser than the fused path's bar: this is the composite, which rounds each intermediate
    # back to bfloat16 in L1, so (1 - s) loses precision the fused kernel keeps in float32 DEST.
    assert_with_pcc(_torch_sigmoid_bw(torch_grad, torch_input), ttnn.to_torch(output), 0.999)


def test_sigmoid_bw_preserves_input_physical_padding(device):
    """An input padded beyond tile alignment must get an output allocated to the same padded
    shape. This is a one-to-one physical-tile kernel -- the factory emits
    input.physical_volume() / TILE_HW pages -- so an output spec that recomputes only the
    minimum tile padding from the logical shape is too small and the writer runs past its end.
    Here a logical 40x40 sits in a 96x96 padded shape (9 tiles) while tile-padding the logical
    shape alone would give 64x64 (4 tiles)."""
    torch.manual_seed(0)
    padded_shape = [1, 1, 96, 96]

    row_major_input = ttnn.from_torch(
        torch.randn((1, 1, 40, 40), dtype=torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    row_major_grad = ttnn.from_torch(
        torch.randn((1, 1, 40, 40), dtype=torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    input_tensor = ttnn.tilize_with_val_padding(row_major_input, padded_shape, 0.0)
    grad_tensor = ttnn.tilize_with_val_padding(row_major_grad, padded_shape, 0.0)
    assert list(input_tensor.padded_shape) == padded_shape, "test setup did not produce extra padding"

    output = ttnn.sigmoid_bw(grad_tensor, input_tensor)[0]

    assert list(output.padded_shape) == list(input_tensor.padded_shape), (
        f"output padded shape {output.padded_shape} does not match the input's "
        f"{input_tensor.padded_shape}; the writer emits one page per input tile, so a smaller "
        "allocation is written past its end"
    )

    # The logical region must still be correct, not merely allocated.
    quantised_input = ttnn.to_torch(input_tensor).float()[..., :40, :40]
    quantised_grad = ttnn.to_torch(grad_tensor).float()[..., :40, :40]
    expected = _torch_sigmoid_bw(quantised_grad, quantised_input)
    torch.testing.assert_close(ttnn.to_torch(output).float()[..., :40, :40], expected, rtol=8e-3, atol=1e-4)


def test_sigmoid_bw_rejects_mixed_layouts(device, expect_error):
    """One program walks all operands either by tile or by row, so a ROW_MAJOR grad_output with a
    TILE input must be rejected rather than read with the wrong page geometry."""
    shape = (1, 1, 32, 32)
    torch_input = torch.randn(shape, dtype=torch.float32)

    tile = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    row_major = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with expect_error(RuntimeError, "same layout"):
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
