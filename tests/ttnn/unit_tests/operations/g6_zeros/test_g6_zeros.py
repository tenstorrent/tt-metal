# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_zeros.

Covers the four production cases and the cases the spec's rules require:
- a dirty destination,
- new storage on every call,
- program-cache reuse across shapes,
- the argument checks.

Every element must read back as the all-zero bit pattern of its dtype. The
comparison is exact on bit patterns, so -0.0 fails. Cases whose dtype is
outside the op's SUPPORTED must be refused with the typed support refusal.
Once refused, they xfail. They run as soon as a refinement adds the dtype.
"""

import pytest
import torch
import ttnn

from ttnn.operations._op_contract import SupportRefusal
from ttnn.operations.g6_zeros import SUPPORTED, g6_zeros

P1 = ((32831, 1, 32, 256), ttnn.bfloat8_b)
P2 = ((16, 12, 128, 128), ttnn.float32)
P3 = ((8, 12, 128, 128), ttnn.float32)
P4 = ((1, 4, 256, 128), ttnn.bfloat16)

PRODUCTION_CASES = [
    pytest.param(*P2, id="p2-float32-16x12x128x128"),
    pytest.param(*P3, id="p3-float32-8x12x128x128"),
    pytest.param(*P1, id="p1-bfloat8_b-32831x1x32x256"),
    pytest.param(*P4, id="p4-bfloat16-1x4x256x128"),
]

_TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}
# The host unpacks bfloat8_b. Widening to float32 keeps the sign bit, so a -0.0 datum still fails.
_COMPARE_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.float32}
_BITS_VIEW = {torch.float32: torch.int32, torch.bfloat16: torch.int16}
# Compare in slices of the leading dim to bound host memory: p1 reads back about 2.7e8 elements.
_LEADING_ROWS_PER_SLICE = 4096


def torch_zeros_reference(shape, dtype):
    return torch.zeros(tuple(shape), dtype=_TORCH_DTYPE[dtype])


def run_g6_zeros(shape, dtype, device):
    return g6_zeros(
        list(shape),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def require_supported(shape, dtype, device, expect_error):
    if dtype in SUPPORTED["dtype"]:
        return
    with expect_error(SupportRefusal, ""):
        run_g6_zeros(shape, dtype, device)
    pytest.xfail(f"{dtype} is outside SUPPORTED and was refused with a SupportRefusal")


def assert_all_zero_bits(output, shape, dtype):
    assert ttnn.is_tensor_storage_on_device(output)
    assert list(output.shape) == list(shape)
    assert output.dtype == dtype
    assert output.layout == ttnn.TILE_LAYOUT
    memory_config = output.memory_config()
    assert memory_config.buffer_type == ttnn.BufferType.DRAM
    assert memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED

    readback = ttnn.to_torch(output)
    assert list(readback.shape) == list(shape)
    compare_dtype = _COMPARE_DTYPE[dtype]
    bits_view = _BITS_VIEW[compare_dtype]
    wrong = 0
    for start in range(0, shape[0], _LEADING_ROWS_PER_SLICE):
        actual = readback[start : start + _LEADING_ROWS_PER_SLICE].to(compare_dtype).contiguous().view(bits_view)
        expected = torch_zeros_reference(actual.shape, dtype).to(compare_dtype).view(bits_view)
        if not torch.equal(actual, expected):
            wrong += int((actual != expected).sum())
    assert wrong == 0, f"{wrong} of {readback.numel()} elements are not the +0.0 bit pattern"


def poison_freed_range(shape, dtype, device):
    """Fill a fresh DRAM tensor of the output's size with non-zero values, free it, return its address."""
    pattern = torch.randn(tuple(shape), dtype=_TORCH_DTYPE[dtype]).abs_().add_(1.0)
    dirty = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(pattern, dtype=dtype, layout=ttnn.TILE_LAYOUT), dirty)
    del pattern
    landed = ttnn.to_torch(dirty)
    assert not bool((landed == 0).any()), "the poison did not land; the case would test nothing"
    del landed
    address = dirty.buffer_address()
    ttnn.deallocate(dirty)
    return address


@pytest.mark.parametrize("shape, dtype", PRODUCTION_CASES)
def test_g6_zeros(shape, dtype, device, expect_error):
    require_supported(shape, dtype, device, expect_error)
    output = run_g6_zeros(shape, dtype, device)
    assert_all_zero_bits(output, shape, dtype)


@pytest.mark.parametrize(
    "shape, dtype",
    [
        pytest.param(*P3, id="p3-float32"),
        pytest.param(*P2, id="p2-float32"),
        pytest.param(*P4, id="p4-bfloat16"),
        pytest.param(*P1, id="p1-bfloat8_b"),
    ],
)
def test_g6_zeros_dirty_destination(shape, dtype, device, expect_error):
    """The output must land on a freed range that held non-zero data, and come out all zero twice."""
    require_supported(shape, dtype, device, expect_error)
    torch.manual_seed(42)

    # The warm call builds and caches the program before any address is pinned.
    warm = run_g6_zeros(shape, dtype, device)
    assert_all_zero_bits(warm, shape, dtype)
    ttnn.deallocate(warm)

    pinned = None
    for round_number in (1, 2):
        address = poison_freed_range(shape, dtype, device)
        if pinned is not None:
            assert address == pinned, f"round {round_number} poisoned {address:#x}, round 1 poisoned {pinned:#x}"
        pinned = address
        output = run_g6_zeros(shape, dtype, device)
        assert output.buffer_address() == address, (
            f"round {round_number}: output at {output.buffer_address():#x}, poisoned range at {address:#x}; "
            "the case cannot tell an op that writes nothing from one that writes zeros"
        )
        assert_all_zero_bits(output, shape, dtype)
        ttnn.deallocate(output)


def test_g6_zeros_new_storage(device, expect_error):
    """Two calls return two buffers; a write into the first never shows in the second."""
    shape, dtype = P3
    require_supported(shape, dtype, device, expect_error)
    first = run_g6_zeros(shape, dtype, device)
    second = run_g6_zeros(shape, dtype, device)
    assert first.buffer_address() != second.buffer_address()
    assert_all_zero_bits(first, shape, dtype)
    assert_all_zero_bits(second, shape, dtype)

    ones = torch.ones(shape, dtype=_TORCH_DTYPE[dtype])
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(ones, dtype=dtype, layout=ttnn.TILE_LAYOUT), first)
    assert bool((ttnn.to_torch(first) == 1).all()), "the write of ones did not land; the case would test nothing"
    assert_all_zero_bits(second, shape, dtype)


def test_g6_zeros_cache_hit_new_address(device, expect_error):
    """A repeat call reuses the cached descriptor; its output address must be re-patched.

    The first output stays alive, so the second call lands on a different, poisoned range. A stale
    address would zero the first buffer again and leave the poison in the second.
    """
    shape, dtype = P3
    require_supported(shape, dtype, device, expect_error)
    torch.manual_seed(7)
    first = run_g6_zeros(shape, dtype, device)
    address = poison_freed_range(shape, dtype, device)
    assert address != first.buffer_address()
    second = run_g6_zeros(shape, dtype, device)
    assert (
        second.buffer_address() == address
    ), f"output at {second.buffer_address():#x}, poisoned range at {address:#x}; the case would test nothing"
    assert_all_zero_bits(second, shape, dtype)
    assert_all_zero_bits(first, shape, dtype)


def test_g6_zeros_program_cache_reuse(device, expect_error):
    """p3, p2, p3 in one process: a repeated shape adds no program-cache entry and still writes zeros."""
    for shape, dtype in (P3, P2):
        require_supported(shape, dtype, device, expect_error)
    device.enable_program_cache()
    seen = set()
    for position, (shape, dtype) in enumerate((P3, P2, P3), start=1):
        before = device.num_program_cache_entries()
        output = run_g6_zeros(shape, dtype, device)
        added = device.num_program_cache_entries() - before
        assert_all_zero_bits(output, shape, dtype)
        ttnn.deallocate(output)
        if shape in seen:
            assert added == 0, f"call {position} repeats {list(shape)} and added {added} program-cache entries"
        seen.add(shape)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"shape": (8, 12, 128)}, id="rank-3"),
        pytest.param({"shape": (8, 12, 128, 128, 1)}, id="rank-5"),
        pytest.param({"shape": (0, 12, 128, 128)}, id="zero-size"),
        pytest.param({"shape": (8, 12, 100, 128)}, id="d2-not-multiple-of-32"),
        pytest.param({"shape": (8, 12, 128, 48)}, id="d3-not-multiple-of-32"),
        pytest.param({"shape": (8, 12, 128.0, 128)}, id="non-integer-size"),
        pytest.param({"layout": ttnn.ROW_MAJOR_LAYOUT}, id="row-major-layout"),
        pytest.param({"memory_config": ttnn.L1_MEMORY_CONFIG}, id="l1-interleaved"),
        pytest.param({"memory_config": ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG}, id="l1-height-sharded"),
        pytest.param(
            {"memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM)},
            id="dram-height-sharded",
        ),
        pytest.param({"device": None}, id="no-device"),
    ],
)
def test_g6_zeros_rejects_invalid_arguments(overrides, device, expect_error):
    kwargs = {
        "shape": P3[0],
        "dtype": ttnn.float32,
        "layout": ttnn.TILE_LAYOUT,
        "device": device,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
    }
    kwargs.update(overrides)
    shape = list(kwargs.pop("shape"))
    with expect_error((ValueError, RuntimeError), ""):
        g6_zeros(shape, **kwargs)
