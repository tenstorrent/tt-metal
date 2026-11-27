# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_zeros (immutable spec for the implementer).

The cases are the four production calls of the golden INPUTS and the cases the
spec's rules require (BB-22), nothing else. A production case whose dtype is not
yet in the op's SUPPORTED must be refused with UnsupportedAxisValue; once
SUPPORTED grows, the same test checks its values. Every output is graded exactly,
on bit patterns: +0.0 everywhere, -0.0 fails (helpers.TOLERANCES is "exact").
"""

import pytest
import torch
import ttnn

from ttnn.operations._op_contract import UnsupportedAxisValue
from ttnn.operations.g6_zeros import SUPPORTED, g6_zeros

PRODUCTION_CASES = [
    pytest.param((16, 12, 128, 128), ttnn.float32, id="p2-16x12x128x128-float32"),
    pytest.param((8, 12, 128, 128), ttnn.float32, id="p3-8x12x128x128-float32"),
    pytest.param((32831, 1, 32, 256), ttnn.bfloat8_b, id="p1-32831x1x32x256-bfloat8_b"),
    pytest.param((1, 4, 256, 128), ttnn.bfloat16, id="p4-1x4x256x128-bfloat16"),
]

# bfloat8_b has no torch dtype: its reference is torch.zeros in bfloat16, whose
# bfloat8_b rendering is exact zero.
_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,
}

_BITS_VIEW = {
    torch.float32: torch.int32,
    torch.bfloat16: torch.int16,
}


def pytorch_g6_zeros(shape, dtype):
    return torch.zeros(tuple(shape), dtype=dtype)


def _call(shape, dtype, device):
    # Production passes the shape as a list literal.
    return g6_zeros(
        list(shape),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _is_supported(dtype):
    return dtype in SUPPORTED["dtype"]


# The spec fixes the exception types only; any message matches.
ANY_MESSAGE = ""


def _expect_refusal(shape, dtype, device, expect_error):
    with expect_error(UnsupportedAxisValue, ANY_MESSAGE):
        _call(shape, dtype, device)


def _bits(tensor):
    return tensor.contiguous().view(_BITS_VIEW[tensor.dtype])


def _check_zeros(output, shape, dtype):
    """Output contract plus the exact all-zero gate on bit patterns."""
    assert ttnn.is_tensor_storage_on_device(output), "output is not on device"
    assert list(output.shape) == list(shape), f"shape {list(output.shape)} != {list(shape)}"
    assert output.dtype == dtype, f"dtype {output.dtype} != {dtype}"
    assert output.layout == ttnn.TILE_LAYOUT, f"layout {output.layout} != TILE"
    memory_config = output.memory_config()
    assert memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, f"placement {memory_config}"
    assert memory_config.buffer_type == ttnn.BufferType.DRAM, f"placement {memory_config}"

    actual = ttnn.to_torch(output)
    expected = pytorch_g6_zeros(shape, _TORCH_DTYPE[dtype])
    if dtype == ttnn.bfloat8_b:
        # The cast keeps the sign of a zero, so a datum read back as -0.0 still fails.
        actual = actual.to(torch.float32)
        expected = expected.to(torch.float32)
    assert actual.dtype == expected.dtype, f"readback dtype {actual.dtype} != {expected.dtype}"
    assert actual.shape == expected.shape, f"readback shape {list(actual.shape)} != {list(expected.shape)}"

    actual_bits, expected_bits = _bits(actual), _bits(expected)
    if not torch.equal(actual_bits, expected_bits):
        wrong = (actual_bits != expected_bits).reshape(-1)
        negative_zero = int((actual_bits == torch.iinfo(actual_bits.dtype).min).sum().item())
        first = int(wrong.nonzero()[0].item())
        pytest.fail(
            f"{int(wrong.sum().item())} of {wrong.numel()} elements are not +0.0 "
            f"({negative_zero} are -0.0); first wrong flat index {first} = {actual.reshape(-1)[first].item()!r}"
        )


def _poison_range(shape, dtype, device):
    """Fill a fresh DRAM-interleaved tensor of the output's size with non-zero
    values, free it, and return the address of the range it held."""
    torch.manual_seed(42)
    # |randn| + 1 lies in [1, ~7): non-zero in every dtype, bfloat8_b included.
    poison = (torch.randn(tuple(shape), dtype=torch.float32).abs() + 1.0).to(_TORCH_DTYPE[dtype])
    dirty = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(poison, dtype=dtype, layout=ttnn.TILE_LAYOUT), dirty)
    del poison
    if bool((ttnn.to_torch(dirty) == 0).any()):
        raise RuntimeError("the poison did not land; the dirty-destination case would test nothing")
    address = dirty.buffer_address()
    ttnn.deallocate(dirty)
    return address


@pytest.mark.parametrize("shape, dtype", PRODUCTION_CASES)
def test_g6_zeros(shape, dtype, device, expect_error):
    if not _is_supported(dtype):
        _expect_refusal(shape, dtype, device, expect_error)
        return
    output = _call(shape, dtype, device)
    _check_zeros(output, shape, dtype)


@pytest.mark.parametrize("shape, dtype", PRODUCTION_CASES)
def test_g6_zeros_dirty_destination(shape, dtype, device, expect_error):
    """The output lands on a range that held non-zero data and still reads all
    zero, twice on the same range: the op must write every page itself."""
    if not _is_supported(dtype):
        _expect_refusal(shape, dtype, device, expect_error)
        return

    # Build and cache the program before pinning an address, so a DRAM allocation
    # the first build makes cannot take the poisoned range ahead of the output.
    warm = _call(shape, dtype, device)
    _check_zeros(warm, shape, dtype)
    ttnn.deallocate(warm)

    pinned = None
    for round_number in (1, 2):
        address = _poison_range(shape, dtype, device)
        if pinned is not None:
            assert address == pinned, f"round {round_number} poisoned {address:#x}, round 1 poisoned {pinned:#x}"
        pinned = address
        output = _call(shape, dtype, device)
        assert output.buffer_address() == address, (
            f"round {round_number}: output at {output.buffer_address():#x}, poisoned range at {address:#x}; "
            "the case cannot tell an op that writes nothing from one that writes zeros"
        )
        _check_zeros(output, shape, dtype)
        ttnn.deallocate(output)


def test_g6_zeros_new_storage(device):
    """Two calls return two buffers; a write into the first never shows in the second."""
    shape, dtype = (8, 12, 128, 128), ttnn.float32
    first = g6_zeros(
        tuple(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    second = _call(shape, dtype, device)
    _check_zeros(first, shape, dtype)
    _check_zeros(second, shape, dtype)
    assert first.buffer_address() != second.buffer_address(), "both calls returned the same buffer"

    ones = torch.ones(shape, dtype=torch.float32)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(ones, dtype=dtype, layout=ttnn.TILE_LAYOUT), first)
    assert bool((ttnn.to_torch(first) == 1).all()), "the write of ones into the first result did not land"
    _check_zeros(second, shape, dtype)


def test_g6_zeros_program_cache(device):
    """p3, p2, p3 in one process: every result is all zero and the repeated
    shape adds no program-cache entry (BB-07)."""
    device.enable_program_cache()
    dtype = ttnn.float32
    seen = set()
    for position, shape in enumerate([(8, 12, 128, 128), (16, 12, 128, 128), (8, 12, 128, 128)], start=1):
        before = device.num_program_cache_entries()
        output = _call(shape, dtype, device)
        added = device.num_program_cache_entries() - before
        _check_zeros(output, shape, dtype)
        ttnn.deallocate(output)
        if shape in seen:
            assert added == 0, f"call {position} repeats {list(shape)} and added {added} program-cache entries"
        seen.add(shape)


INVALID_ARGUMENTS = [
    pytest.param({"shape": (8, 12, 128)}, id="rank3"),
    pytest.param({"shape": (1, 8, 12, 128, 128)}, id="rank5"),
    pytest.param({"shape": (0, 12, 128, 128)}, id="zero-size"),
    pytest.param({"shape": (8, 12, 128, 48)}, id="width-not-multiple-of-32"),
    pytest.param({"shape": (8, 12, 100, 128)}, id="height-not-multiple-of-32"),
    pytest.param({"shape": (8, 12, 128.0, 128)}, id="non-integer-size"),
    pytest.param({"layout": ttnn.ROW_MAJOR_LAYOUT}, id="row-major-layout"),
    pytest.param({"memory_config": ttnn.L1_MEMORY_CONFIG}, id="l1-interleaved"),
    pytest.param({"memory_config": ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG}, id="l1-sharded"),
    pytest.param(
        {"memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM)},
        id="dram-sharded",
    ),
    pytest.param({"device": None}, id="no-device"),
]


@pytest.mark.parametrize("overrides", INVALID_ARGUMENTS)
def test_g6_zeros_rejects_invalid_arguments(overrides, device, expect_error):
    kwargs = {
        "shape": (8, 12, 128, 128),
        "dtype": ttnn.float32,
        "layout": ttnn.TILE_LAYOUT,
        "device": device,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
    }
    kwargs.update(overrides)
    shape = kwargs.pop("shape")
    with expect_error((ValueError, RuntimeError), ANY_MESSAGE):
        g6_zeros(shape, **kwargs)
