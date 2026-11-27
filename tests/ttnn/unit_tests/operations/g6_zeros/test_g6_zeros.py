# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_zeros: the specification the implementation must pass.

Only the production cases of the golden INPUTS and the cases the spec's rules
require are tested (BB-22). A case whose dtype is outside the op's SUPPORTED
must be refused with a SupportRefusal (a NotImplementedError) and is then
reported as xfail; once a refinement adds the dtype, the same test runs the
full check unchanged.

The reference is torch.zeros. The gate is exact on bit patterns (the op's
helpers.TOLERANCES entry is ("exact", None) for every dtype): -0.0 fails
although it equals zero. bfloat8_b is compared after the host unpacks it to
float32, a cast that keeps the sign of a zero.
"""

import inspect

import pytest
import torch
import ttnn

from ttnn.operations.g6_zeros import g6_zeros
from ttnn.operations.g6_zeros import EXCLUSIONS, INPUT_TAGGERS, SUPPORTED

# Production cases of eval/golden_tests/g6_zeros/feature_spec.py INPUTS.
P1 = ((32831, 1, 32, 256), ttnn.bfloat8_b)  # 262648 pages, the largest output
P2 = ((16, 12, 128, 128), ttnn.float32)  # 3072 pages
P3 = ((8, 12, 128, 128), ttnn.float32)  # 1536 pages
P4 = ((1, 4, 256, 128), ttnn.bfloat16)  # 128 pages, the most called

_HOST_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}
_BITS_VIEW = {torch.float32: torch.int32, torch.bfloat16: torch.int16}
_POISON_BITS = {torch.float32: 0x7F7F7F7F, torch.bfloat16: 0x7F7F}

# Bounds the host memory of a bit comparison on p1 (268951552 elements).
_COMPARE_CHUNK_ELEMENTS = 1 << 24


def _param(name, case):
    shape, dtype = case
    return pytest.param(shape, dtype, id=f"{name}-{'x'.join(str(d) for d in shape)}-{dtype}")


PRODUCTION_CASES = [_param("p1", P1), _param("p2", P2), _param("p3", P3), _param("p4", P4)]


def torch_g6_zeros(shape, dtype):
    """PyTorch reference: torch.zeros in the host dtype that renders `dtype`."""
    return torch.zeros(tuple(shape), dtype=_HOST_DTYPE[dtype])


def _call(shape, dtype, device):
    # Production passes the shape as a list literal and all four keywords.
    return g6_zeros(
        list(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _is_supported(shape, dtype):
    scenario = {"shape": tuple(shape), "dtype": dtype, "layout": ttnn.TILE_LAYOUT}
    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger((scenario,), axes)
    if any(axes.get(axis) not in allowed for axis, allowed in SUPPORTED.items()):
        return False
    return not any(all(axes.get(key) == value for key, value in cell.items()) for cell in EXCLUSIONS)


def _skip_unless_supported(shape, dtype, device, expect_error):
    """Outside SUPPORTED the op must refuse with a SupportRefusal; the case is then xfail."""
    if _is_supported(shape, dtype):
        return
    # The refusal text is free-form (ttnn.operations._op_contract), so any message matches.
    with expect_error(NotImplementedError, ""):
        _call(shape, dtype, device)
    pytest.xfail(f"{dtype} is outside SUPPORTED {SUPPORTED}; refused as required")


def _assert_output_contract(output, shape, dtype):
    assert ttnn.is_tensor_storage_on_device(output), "output is not on device"
    assert list(output.shape) == list(shape), f"shape {list(output.shape)} != {list(shape)}"
    assert output.dtype == dtype, f"dtype {output.dtype} != {dtype}"
    assert output.layout == ttnn.TILE_LAYOUT, f"layout {output.layout} != TILE"
    memory_config = output.memory_config()
    assert memory_config.buffer_type == ttnn.BufferType.DRAM, f"placement {memory_config}"
    assert memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, f"placement {memory_config}"


def _readback(device_tensor, dtype):
    readback = ttnn.to_torch(device_tensor)
    if dtype == ttnn.bfloat8_b:
        readback = readback.to(torch.float32)
    return readback


def _assert_all_zero_bits(output, shape, dtype):
    """Every device tensor of `output` equals the reference bit for bit.

    Returns the bit patterns of the first device tensor, for bit-identity checks.
    """
    _assert_output_contract(output, shape, dtype)
    first_bits = None
    for device_index, device_tensor in enumerate(ttnn.get_device_tensors(output)):
        readback = _readback(device_tensor, dtype)
        assert list(readback.shape) == list(shape), f"device {device_index}: readback shape {list(readback.shape)}"
        bits = readback.contiguous().view(_BITS_VIEW[readback.dtype])
        row_elements = max(1, bits[0].numel())
        rows_per_chunk = max(1, _COMPARE_CHUNK_ELEMENTS // row_elements)
        for chunk_start in range(0, shape[0], rows_per_chunk):
            actual = bits[chunk_start : chunk_start + rows_per_chunk]
            expected = torch_g6_zeros(actual.shape, dtype).to(readback.dtype)
            expected_bits = expected.view(_BITS_VIEW[expected.dtype])
            if torch.equal(actual, expected_bits):
                continue
            wrong = actual != expected_bits
            negative_zero = int((actual == torch.iinfo(actual.dtype).min).sum().item())
            first = tuple(int(i) for i in wrong.nonzero()[0])
            first = (first[0] + chunk_start,) + first[1:]
            pytest.fail(
                f"device {device_index}: {int(wrong.sum().item())} elements of rows "
                f"[{chunk_start}, {chunk_start + actual.shape[0]}) are not +0.0 bits "
                f"({negative_zero} of them -0.0); first at {first} = {readback[first].item()!r}"
            )
        if first_bits is None:
            first_bits = bits
    return first_bits


def _poison_range(shape, dtype, device):
    """Fill a fresh DRAM-interleaved tensor of the output's size with a loud
    non-zero pattern, free it, and return the address of the range it held."""
    host_dtype = _HOST_DTYPE[dtype]
    pattern = torch.full(tuple(shape), _POISON_BITS[host_dtype], dtype=_BITS_VIEW[host_dtype]).view(host_dtype)
    dirty = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(pattern, dtype=dtype, layout=ttnn.TILE_LAYOUT), dirty)
    del pattern
    if bool((ttnn.to_torch(dirty) == 0).any()):
        raise RuntimeError("the poison pattern did not land; the dirty-destination case would test nothing")
    address = dirty.buffer_address()
    ttnn.deallocate(dirty)
    return address


@pytest.mark.parametrize("shape, dtype", PRODUCTION_CASES)
def test_g6_zeros_production(shape, dtype, device, expect_error):
    """Every production case: output contract and all-zero bits on every device."""
    _skip_unless_supported(shape, dtype, device, expect_error)
    output = _call(shape, dtype, device)
    _assert_all_zero_bits(output, shape, dtype)


@pytest.mark.parametrize("shape, dtype", PRODUCTION_CASES)
def test_g6_zeros_dirty_destination(shape, dtype, device, expect_error):
    """Spec rule 1: zeros land on a range that held a loud pattern, twice, bit-identical (BB-06).

    A warm call builds the program before any address is pinned. Each round poisons
    the range, frees it and requires the output at exactly that address: an
    unconfirmed reuse fails, because an op that writes nothing passes on a clean range.
    """
    _skip_unless_supported(shape, dtype, device, expect_error)
    warm = _call(shape, dtype, device)
    _assert_all_zero_bits(warm, shape, dtype)
    ttnn.deallocate(warm)

    results, pinned = [], None
    for round_number in (1, 2):
        address = _poison_range(shape, dtype, device)
        assert (
            pinned is None or address == pinned
        ), f"round {round_number} poisoned {address:#x}, not the range of round 1 ({pinned:#x})"
        pinned = address
        output = _call(shape, dtype, device)
        assert output.buffer_address() == address, (
            f"round {round_number}: output at {output.buffer_address():#x}, poisoned range at {address:#x}; "
            f"the case cannot tell an op that writes nothing from one that writes zeros"
        )
        results.append(_assert_all_zero_bits(output, shape, dtype))
        ttnn.deallocate(output)
    assert torch.equal(results[0], results[1]), "the two dirty-destination results differ"


def test_g6_zeros_new_storage(device, expect_error):
    """Spec rule 2: two calls return two buffers; a write into the first never shows in the second."""
    shape, dtype = P3
    _skip_unless_supported(shape, dtype, device, expect_error)
    first = _call(shape, dtype, device)
    second = _call(shape, dtype, device)
    _assert_all_zero_bits(first, shape, dtype)
    _assert_all_zero_bits(second, shape, dtype)
    assert first.buffer_address() != second.buffer_address(), "both calls returned the same buffer"
    ones = ttnn.from_torch(torch.ones(shape, dtype=_HOST_DTYPE[dtype]), dtype=dtype, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(ones, first)
    if not bool((ttnn.to_torch(first) == 1).all()):
        raise RuntimeError("the write of ones into the first result did not land; the case would test nothing")
    _assert_all_zero_bits(second, shape, dtype)


def test_g6_zeros_program_cache(device, expect_error):
    """BB-07: p3, p2, p3 in one process; every result is right and a repeated shape adds no cache entry."""
    dtype = ttnn.float32
    sequence = (P3[0], P2[0], P3[0])
    _skip_unless_supported(sequence[0], dtype, device, expect_error)
    device.enable_program_cache()
    seen = set()
    for position, shape in enumerate(sequence, start=1):
        before = device.num_program_cache_entries()
        output = _call(shape, dtype, device)
        added = device.num_program_cache_entries() - before
        _assert_all_zero_bits(output, shape, dtype)
        ttnn.deallocate(output)
        assert not (
            shape in seen and added
        ), f"call {position} repeats {list(shape)} and added {added} program cache entries"
        seen.add(shape)


_INVALID_CALLS = [
    pytest.param({"shape": (12, 128, 128)}, id="rank-3"),
    pytest.param({"shape": (8, 12, 128, 128, 1)}, id="rank-5"),
    pytest.param({"shape": (0, 12, 128, 128)}, id="zero-size"),
    pytest.param({"shape": (8, -12, 128, 128)}, id="negative-size"),
    pytest.param({"shape": (8, 12, 128.0, 128)}, id="non-integer-size"),
    pytest.param({"shape": (8, 12, 100, 128)}, id="d2-not-multiple-of-32"),
    pytest.param({"shape": (8, 12, 128, 100)}, id="d3-not-multiple-of-32"),
    pytest.param({"layout": ttnn.ROW_MAJOR_LAYOUT}, id="row-major-layout"),
    pytest.param({"memory_config": ttnn.L1_MEMORY_CONFIG}, id="l1-interleaved"),
    pytest.param({"memory_config": ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG}, id="l1-height-sharded"),
    pytest.param(
        {"memory_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM)},
        id="dram-width-sharded",
    ),
    pytest.param({"device": None}, id="no-device"),
]


@pytest.mark.parametrize("override", _INVALID_CALLS)
def test_g6_zeros_rejects_invalid_arguments(override, device, expect_error):
    """Validation: each of these raises ValueError or RuntimeError (not a support refusal)."""
    kwargs = {
        "shape": list(P3[0]),
        "dtype": ttnn.float32,
        "layout": ttnn.TILE_LAYOUT,
        "device": device,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
    }
    kwargs.update(override)
    shape = kwargs.pop("shape")
    with expect_error((ValueError, RuntimeError), ""):
        g6_zeros(shape, **kwargs)


def test_g6_zeros_signature():
    """`shape` is positional; dtype, layout, device and memory_config are keyword-only without defaults."""
    parameters = list(inspect.signature(g6_zeros).parameters.values())
    assert [p.name for p in parameters] == ["shape", "dtype", "layout", "device", "memory_config"]
    assert parameters[0].kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    for parameter in parameters[1:]:
        assert parameter.kind == inspect.Parameter.KEYWORD_ONLY, f"{parameter.name} is not keyword-only"
        assert parameter.default is inspect.Parameter.empty, f"{parameter.name} has a default"
