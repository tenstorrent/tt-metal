# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_zeros (immutable spec — do not modify).

Cases are the four production cases of the golden INPUTS plus the cases the
spec's rules require (BB-22): dirty destination, new storage, repeat/cache
order, validation. The gate is exact on bit patterns (helpers.TOLERANCES is
("exact", None) for every dtype): -0.0 fails.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_zeros import g6_zeros
from ttnn.operations._op_contract import SupportRefusal
from eval.golden_tests.g6_zeros.helpers import TOLERANCES

torch.manual_seed(42)

# (shape, dtype): the production cases (feature_spec.INPUTS).
PRODUCTION_CASES = [
    pytest.param((16, 12, 128, 128), ttnn.float32, id="p2_fp32_16x12x128x128"),
    pytest.param((8, 12, 128, 128), ttnn.float32, id="p3_fp32_8x12x128x128"),
    pytest.param((32831, 1, 32, 256), ttnn.bfloat8_b, id="p1_bfp8_32831x1x32x256"),
    pytest.param((1, 4, 256, 128), ttnn.bfloat16, id="p4_bf16_1x4x256x128"),
]

_TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}
_BITS = {torch.float32: torch.int32, torch.bfloat16: torch.int16}


def pytorch_zeros(shape, dtype):
    """PyTorch reference."""
    return torch.zeros(tuple(shape), dtype=dtype)


def _call(shape, dtype, device):
    return g6_zeros(
        list(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _call_or_xfail(shape, dtype, device):
    try:
        return _call(shape, dtype, device)
    except SupportRefusal as exc:
        pytest.xfail(f"not in SUPPORTED yet: {exc}")


def _assert_zero_bits(output, shape, dtype):
    assert TOLERANCES[dtype][0] == "exact"
    assert list(output.shape) == list(shape)
    assert output.dtype == dtype
    assert output.layout == ttnn.TILE_LAYOUT
    mc = output.memory_config()
    assert mc.buffer_type == ttnn.BufferType.DRAM
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED

    actual = ttnn.to_torch(output)
    if dtype == ttnn.bfloat8_b:
        actual = actual.to(torch.float32)  # keeps the sign of zero
        expected = pytorch_zeros(shape, torch.bfloat16).to(torch.float32)
    else:
        expected = pytorch_zeros(shape, _TORCH_DTYPE[dtype])
    assert actual.dtype == expected.dtype and list(actual.shape) == list(shape)
    bits = _BITS[actual.dtype]
    actual_bits = actual.contiguous().view(bits)
    expected_bits = expected.contiguous().view(bits)
    wrong = int((actual_bits != expected_bits).sum().item())
    assert wrong == 0, f"{wrong} of {actual.numel()} elements are not the +0.0 bit pattern"
    return actual_bits


@pytest.mark.parametrize("shape, dtype", PRODUCTION_CASES)
def test_g6_zeros(device, shape, dtype):
    output = _call_or_xfail(shape, dtype, device)
    _assert_zero_bits(output, shape, dtype)


@pytest.mark.parametrize("shape, dtype", PRODUCTION_CASES)
def test_g6_zeros_dirty_destination(device, shape, dtype):
    """Spec rule 1: the output lands on a freed range that held a loud non-zero
    pattern and must still come out all zero."""
    warm = _call_or_xfail(shape, dtype, device)  # builds the program first
    ttnn.deallocate(warm)

    torch_dtype = _TORCH_DTYPE[dtype]
    poison = torch.full(
        tuple(shape), 0x3F80 if torch_dtype == torch.bfloat16 else 0x3F800000, dtype=_BITS[torch_dtype]
    ).view(
        torch_dtype
    )  # 1.0
    dirty = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(poison, dtype=dtype, layout=ttnn.TILE_LAYOUT), dirty)
    assert bool((ttnn.to_torch(dirty) != 0).all()), "poison did not land"
    address = dirty.buffer_address()
    ttnn.deallocate(dirty)

    output = _call(shape, dtype, device)
    assert output.buffer_address() == address, "output did not reuse the poisoned range; case is vacuous"
    _assert_zero_bits(output, shape, dtype)


@pytest.mark.parametrize("shape, dtype", PRODUCTION_CASES[1:2])
def test_g6_zeros_new_storage(device, shape, dtype):
    """Spec rule 2: every call returns new storage."""
    first = _call(shape, dtype, device)
    second = _call(shape, dtype, device)
    assert first.buffer_address() != second.buffer_address()
    ones = torch.ones(tuple(shape), dtype=_TORCH_DTYPE[dtype])
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(ones, dtype=dtype, layout=ttnn.TILE_LAYOUT), first)
    _assert_zero_bits(second, shape, dtype)


def test_g6_zeros_cache_order_and_repeat(device):
    """BB-06 / BB-07: p3, p2, p3 in one process; the repeat adds no program and
    every result is bit-identical zeros."""
    device.enable_program_cache()
    shapes = [(8, 12, 128, 128), (16, 12, 128, 128), (8, 12, 128, 128)]
    seen, reference = set(), {}
    for shape in shapes:
        before = device.num_program_cache_entries()
        output = _call(shape, ttnn.float32, device)
        added = device.num_program_cache_entries() - before
        bits = _assert_zero_bits(output, shape, ttnn.float32)
        ttnn.deallocate(output)
        if shape in seen:
            assert added == 0, f"repeat of {shape} rebuilt the program"
            assert torch.equal(bits, reference[shape])
        seen.add(shape)
        reference[shape] = bits


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"shape": (8, 12, 128)}, id="rank3"),
        pytest.param({"shape": (8, 12, 128, 0)}, id="zero_dim"),
        pytest.param({"shape": (8, 12, 100, 128)}, id="d2_not_mult_32"),
        pytest.param({"shape": (8, 12, 128, 48)}, id="d3_not_mult_32"),
        pytest.param({"layout": ttnn.ROW_MAJOR_LAYOUT}, id="row_major"),
        pytest.param({"memory_config": ttnn.L1_MEMORY_CONFIG}, id="l1"),
        pytest.param({"device": None}, id="no_device"),
    ],
)
def test_g6_zeros_validation(device, kwargs, expect_error):
    args = {
        "shape": (8, 12, 128, 128),
        "dtype": ttnn.float32,
        "layout": ttnn.TILE_LAYOUT,
        "device": device,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
    }
    args.update(kwargs)
    shape = args.pop("shape")
    with expect_error((ValueError, RuntimeError), ""):  # any message: the op words its own refusal
        g6_zeros(list(shape), **args)
