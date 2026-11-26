# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_zeros.

The shapes are the four production cases of the golden INPUTS and nothing else (BB-22): no
single-tile or synthetic shape is tested. A case whose dtype is outside the op's SUPPORTED must be
refused with a SupportRefusal; once a refinement adds that dtype, the same case must produce zeros.
Every result is graded exactly, on bit patterns, against torch.zeros, so -0.0 fails.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_zeros import g6_zeros
from ttnn.operations.g6_zeros import SUPPORTED
from ttnn.operations._op_contract import SupportRefusal

P1 = (32831, 1, 32, 256)
P2 = (16, 12, 128, 128)
P3 = (8, 12, 128, 128)
P4 = (1, 4, 256, 128)

PRODUCTION_CASES = [
    pytest.param(P2, ttnn.float32, id="p2-16x12x128x128-float32"),
    pytest.param(P3, ttnn.float32, id="p3-8x12x128x128-float32"),
    pytest.param(P4, ttnn.bfloat16, id="p4-1x4x256x128-bfloat16"),
    pytest.param(P1, ttnn.bfloat8_b, id="p1-32831x1x32x256-bfloat8_b"),
]

# p1 needs a host poison of 269M elements; its dirty-destination case stays with the golden suite.
DIRTY_DESTINATION_CASES = PRODUCTION_CASES[:3]

# bfloat8_b has no torch dtype: its reference is torch.zeros in bfloat16 and its readback is compared
# after the host unpack, cast to float32 (the cast keeps the sign of a zero).
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}
COMPARE_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.float32}
BITS_VIEW = {torch.float32: torch.int32, torch.bfloat16: torch.int16}

# The spec fixes the exception types of a refusal or a rejected argument, not their wording.
ANY_MESSAGE = ""


def pytorch_reference(shape, dtype):
    return torch.zeros(tuple(shape), dtype=dtype)


def call_op(shape, dtype, device):
    # Production passes the shape as a list.
    return g6_zeros(
        list(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def supported_or_refused(shape, dtype, device, expect_error):
    """True when the op claims `dtype`; otherwise requires the call to be refused and returns False."""
    if dtype in SUPPORTED["dtype"]:
        return True
    with expect_error(SupportRefusal, ANY_MESSAGE):
        call_op(shape, dtype, device)
    return False


def bit_patterns(tensor):
    return tensor.contiguous().view(BITS_VIEW[tensor.dtype])


def check_zeros(output, shape, dtype):
    """Output contract plus the exact gate. Returns the output's bit patterns."""
    assert ttnn.is_tensor_storage_on_device(output)
    assert list(output.shape) == list(shape)
    assert output.dtype == dtype
    assert output.layout == ttnn.TILE_LAYOUT
    memory_config = output.memory_config()
    assert memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert memory_config.buffer_type == ttnn.BufferType.DRAM

    readback = ttnn.to_torch(output).to(COMPARE_DTYPE[dtype])
    expected = pytorch_reference(shape, TORCH_DTYPE[dtype]).to(COMPARE_DTYPE[dtype])
    assert readback.shape == expected.shape
    actual_bits, expected_bits = bit_patterns(readback), bit_patterns(expected)
    if not torch.equal(actual_bits, expected_bits):
        wrong = actual_bits != expected_bits
        negative_zero = int((actual_bits == torch.iinfo(actual_bits.dtype).min).sum())
        pytest.fail(f"{int(wrong.sum())} of {wrong.numel()} elements are not +0.0 ({negative_zero} of them are -0.0)")
    return actual_bits


def poison_range(shape, dtype, device):
    """Fill a fresh DRAM-interleaved range of the output's size with non-zero values, free it, and
    return its address. One device allocation only, so the freed range is the only hole it leaves."""
    poison = (torch.randn(tuple(shape), dtype=torch.float32).abs() + 1.0).to(TORCH_DTYPE[dtype])
    dirty = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(poison, dtype=dtype, layout=ttnn.TILE_LAYOUT), dirty)
    assert not bool((ttnn.to_torch(dirty) == 0).any()), "the poison did not land; the case would test nothing"
    address = dirty.buffer_address()
    ttnn.deallocate(dirty)
    return address


@pytest.mark.parametrize("shape, dtype", PRODUCTION_CASES)
def test_production_case(shape, dtype, device, expect_error):
    if not supported_or_refused(shape, dtype, device, expect_error):
        return
    check_zeros(call_op(shape, dtype, device), shape, dtype)


@pytest.mark.parametrize("shape, dtype", DIRTY_DESTINATION_CASES)
def test_dirty_destination(shape, dtype, device, expect_error):
    """Every page is written over stale data, on the exact poisoned range, twice, bit-identically."""
    if not supported_or_refused(shape, dtype, device, expect_error):
        return
    torch.manual_seed(42)
    # The first call builds the program, so DRAM the build takes cannot land on the poisoned range.
    warm = call_op(shape, dtype, device)
    check_zeros(warm, shape, dtype)
    ttnn.deallocate(warm)

    results = []
    for _ in range(2):
        address = poison_range(shape, dtype, device)
        output = call_op(shape, dtype, device)
        assert output.buffer_address() == address, (
            f"output at {output.buffer_address():#x}, poisoned range at {address:#x}: "
            "the case cannot tell an op that writes nothing from one that writes zeros"
        )
        results.append(check_zeros(output, shape, dtype))
        ttnn.deallocate(output)
    assert torch.equal(results[0], results[1])


def test_program_cache(device, expect_error):
    """p3, p2, p3 in one process: every result exact, and the repeated shape adds no cache entry."""
    dtype = ttnn.float32
    if not supported_or_refused(P3, dtype, device, expect_error):
        return
    device.enable_program_cache()
    seen = set()
    for shape in (P3, P2, P3):
        before = device.num_program_cache_entries()
        output = call_op(shape, dtype, device)
        added = device.num_program_cache_entries() - before
        check_zeros(output, shape, dtype)
        ttnn.deallocate(output)
        if shape in seen:
            assert added == 0, f"the repeat of {shape} added {added} program cache entries"
        seen.add(shape)


def test_new_storage(device, expect_error):
    """Two calls return two buffers; ones written into the first never show in the second."""
    shape, dtype = P3, ttnn.float32
    if not supported_or_refused(shape, dtype, device, expect_error):
        return
    first = call_op(shape, dtype, device)
    second = call_op(shape, dtype, device)
    check_zeros(first, shape, dtype)
    check_zeros(second, shape, dtype)
    assert first.buffer_address() != second.buffer_address()

    ones = ttnn.from_torch(torch.ones(shape, dtype=TORCH_DTYPE[dtype]), dtype=dtype, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(ones, first)
    assert bool((ttnn.to_torch(first) == 1).all()), "the write of ones did not land; the case would test nothing"
    check_zeros(second, shape, dtype)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((8, 128, 128), id="rank3"),
        pytest.param((1, 8, 12, 128, 128), id="rank5"),
        pytest.param((1, 1, 48, 64), id="d2-not-multiple-of-32"),
        pytest.param((1, 1, 64, 48), id="d3-not-multiple-of-32"),
        pytest.param((0, 1, 32, 32), id="zero-size"),
        pytest.param((-1, 1, 32, 32), id="negative-size"),
        pytest.param((1.0, 1, 32, 32), id="float-size"),
        pytest.param("1x1x32x32", id="not-a-sequence"),
    ],
)
def test_rejects_bad_shape(shape, device, expect_error):
    with expect_error((ValueError, RuntimeError), ANY_MESSAGE):
        g6_zeros(
            shape, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )


def test_rejects_row_major_layout(device, expect_error):
    with expect_error((ValueError, RuntimeError), ANY_MESSAGE):
        g6_zeros(
            list(P3),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


@pytest.mark.parametrize(
    "memory_layout, buffer_type",
    [
        pytest.param(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, id="l1-interleaved"),
        pytest.param(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, id="dram-height-sharded"),
        pytest.param(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, id="l1-height-sharded"),
    ],
)
def test_rejects_memory_config(memory_layout, buffer_type, device, expect_error):
    with expect_error((ValueError, RuntimeError), ANY_MESSAGE):
        g6_zeros(
            list(P3),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(memory_layout, buffer_type),
        )


@pytest.mark.parametrize("not_a_device", [None, "device"], ids=["none", "string"])
def test_rejects_non_device(not_a_device, expect_error):
    with expect_error((ValueError, RuntimeError), ANY_MESSAGE):
        g6_zeros(
            list(P3),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=not_a_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
