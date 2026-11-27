# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_untilize.

Covers the production case (1, 1, 32, 128) uint32 TILE in DRAM and the cases the spec's rules
require (BB-22), nothing else. The op moves bytes, so every comparison is on 32-bit patterns.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_untilize import describe_plan, g6_untilize

SHAPE = (1, 1, 32, 128)
PRODUCTION_KWARGS = {"use_multicore": True, "sub_core_grids": None}


def torch_g6_untilize(x):
    """Reference: the logical values in new storage, in the input dtype."""
    return x.clone()


def _uint32_from_patterns(patterns):
    v = patterns.to(torch.int64) & 0xFFFFFFFF
    v = torch.where(v >= 2**31, v - 2**32, v)
    return v.to(torch.int32).view(torch.uint32)


def _random_uint32(shape):
    # Uniform over all 2^32 patterns: values above 2^24 and 2^31 expose any narrower path.
    return _uint32_from_patterns(torch.randint(-(2**31), 2**31, shape, dtype=torch.int64))


def _bits(t):
    if t.dtype == torch.uint32:
        return t.contiguous().view(torch.int32)
    if t.dtype == torch.int32:
        return t
    if t.dtype == torch.int64:
        return _uint32_from_patterns(t).view(torch.int32)
    raise AssertionError(f"readback dtype {t.dtype} holds no 32-bit integer pattern")


def _to_device(x, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(x, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


def _assert_bits_equal(actual, expected, what):
    assert list(actual.shape) == list(expected.shape), f"{what}: shape {list(actual.shape)} != {list(expected.shape)}"
    a, e = _bits(actual), _bits(expected)
    bad = a != e
    if bool(bad.any()):
        idx = torch.nonzero(bad.reshape(-1)).reshape(-1)[:5].tolist()
        av = a.reshape(-1).to(torch.int64) & 0xFFFFFFFF
        ev = e.reshape(-1).to(torch.int64) & 0xFFFFFFFF
        detail = ", ".join(f"[{i}] got {av[i].item():#010x} expected {ev[i].item():#010x}" for i in idx)
        raise AssertionError(f"{what}: {int(bad.sum())}/{bad.numel()} elements differ; {detail}")


def _check_output(out, expected):
    """Output contract (device, shape, dtype, ROW_MAJOR, DRAM interleaved) and exact values."""
    assert ttnn.is_tensor_storage_on_device(out), "output is not a device tensor"
    assert list(out.shape) == list(SHAPE), f"shape {list(out.shape)} != {list(SHAPE)}"
    assert out.dtype == ttnn.uint32, f"dtype {out.dtype} != uint32"
    assert out.layout == ttnn.ROW_MAJOR_LAYOUT, f"layout {out.layout} != ROW_MAJOR"
    mc = out.memory_config()
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, f"memory layout {mc.memory_layout}"
    assert mc.buffer_type == ttnn.BufferType.DRAM, f"buffer type {mc.buffer_type}"
    actual = ttnn.to_torch(out)
    _assert_bits_equal(actual, expected, "g6_untilize vs torch.clone")
    return actual


@pytest.mark.parametrize("call", ["explicit_keywords", "default_keywords"])
def test_production_case(device, call):
    torch.manual_seed(42)
    x = _random_uint32(SHAPE)
    expected = torch_g6_untilize(x)
    x_tt = _to_device(x, device)
    if call == "explicit_keywords":
        out = g6_untilize(x_tt, **PRODUCTION_KWARGS)
    else:
        out = g6_untilize(x_tt)
    _check_output(out, expected)


def test_face_order_and_extremes(device):
    """Each element holds its own row-major index (+2^31 on odd indices), so a wrong face, tile or
    row order names the misplaced element. Columns 0-3 of every row hold 0, 2^32-1, 2^31, 2^24+1."""
    index = torch.arange(SHAPE[2] * SHAPE[3], dtype=torch.int64).reshape(SHAPE)
    patterns = index + (index % 2) * 2**31
    patterns[..., :4] = torch.tensor([0, 2**32 - 1, 2**31, 2**24 + 1], dtype=torch.int64)
    x = _uint32_from_patterns(patterns)
    out = g6_untilize(_to_device(x, device), **PRODUCTION_KWARGS)
    _check_output(out, torch_g6_untilize(x))


def test_new_storage_input_unchanged(device):
    torch.manual_seed(42)
    x = _random_uint32(SHAPE)
    x_tt = _to_device(x, device)
    out = g6_untilize(x_tt, **PRODUCTION_KWARGS)
    assert out.buffer_address() != x_tt.buffer_address(), "output shares the input's buffer"
    _check_output(out, torch_g6_untilize(x))
    _assert_bits_equal(ttnn.to_torch(x_tt), x, "input changed by the call")


def test_repeat_and_program_cache(device):
    """A second call on new buffers reuses the program and still writes the right buffer; a repeat
    on the first input is bit-identical to the first result."""
    torch.manual_seed(42)
    x1 = _random_uint32(SHAPE)
    x2 = _random_uint32(SHAPE)
    # Both inputs exist before the first op call so that only op dispatches move the cache count.
    x1_tt = _to_device(x1, device)
    x2_tt = _to_device(x2, device)

    out1 = g6_untilize(x1_tt, **PRODUCTION_KWARGS)
    entries = device.num_program_cache_entries()
    out2 = g6_untilize(x2_tt, **PRODUCTION_KWARGS)
    out3 = g6_untilize(x1_tt, **PRODUCTION_KWARGS)
    assert device.num_program_cache_entries() == entries, "a repeat call with the same shape built a new program"

    first = _check_output(out1, torch_g6_untilize(x1))
    _check_output(out2, torch_g6_untilize(x2))
    third = _check_output(out3, torch_g6_untilize(x1))
    _assert_bits_equal(third, first, "repeat call differs from the first")


def test_plan_single_rectangle_multicore(device):
    torch.manual_seed(42)
    x = _random_uint32(SHAPE)
    x_tt = _to_device(x, device)

    plan = describe_plan(x_tt)
    assert plan["regime"] == "row_gather", f"regime {plan['regime']!r}"
    ranges = plan["core_ranges"].ranges()
    assert len(ranges) == 1, f"kernel group spans {len(ranges)} core ranges, expected one rectangle"
    assert plan["num_cores"] == plan["core_ranges"].num_cores()
    assert plan["num_cores"] >= 2, "the production case runs on one core"

    out = g6_untilize(x_tt, **PRODUCTION_KWARGS)
    _check_output(out, torch_g6_untilize(x))

    again = describe_plan(x_tt)
    assert again["regime"] == plan["regime"]
    assert again["num_cores"] == plan["num_cores"]
    assert list(again["core_ranges"].ranges()) == list(ranges)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"use_multicore": False}, "(?i)use_multicore"),
        (
            {"sub_core_grids": ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])},
            "(?i)sub_core_grids",
        ),
    ],
    ids=["use_multicore_false", "sub_core_grids_set"],
)
def test_rejects_keywords(device, expect_error, kwargs, message):
    torch.manual_seed(42)
    x_tt = _to_device(_random_uint32(SHAPE), device)
    with expect_error(ValueError, message):
        g6_untilize(x_tt, **kwargs)


@pytest.mark.parametrize(
    "case, message",
    [("host_tensor", "(?i)device"), ("rank_3", "(?i)rank"), ("l1_interleaved", "(?i)dram")],
    ids=["host_tensor", "rank_3", "l1_interleaved"],
)
def test_rejects_inputs(device, expect_error, case, message):
    torch.manual_seed(42)
    if case == "host_tensor":
        x_tt = ttnn.from_torch(_random_uint32(SHAPE), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)
    elif case == "rank_3":
        x_tt = _to_device(_random_uint32(SHAPE[1:]), device)
    else:
        x_tt = _to_device(_random_uint32(SHAPE), device, memory_config=ttnn.L1_MEMORY_CONFIG)
    with expect_error((ValueError, RuntimeError), message):
        g6_untilize(x_tt, **PRODUCTION_KWARGS)
