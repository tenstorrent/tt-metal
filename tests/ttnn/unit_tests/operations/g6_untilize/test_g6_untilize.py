# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_untilize: uint32 TILE -> ROW_MAJOR, every element bit-identical.

Scope is the production case (1, 1, 32, 128) and the cases the spec's rules require (BB-22).
Inputs are full-range uint32 bit patterns: small integers survive a float32 or 16-bit path.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_untilize import g6_untilize

PRODUCTION_SHAPE = (1, 1, 32, 128)
PRODUCTION_KWARGS = {"use_multicore": True, "sub_core_grids": None}


def torch_reference(x):
    return x.clone()


def uint32_from_patterns(patterns):
    v = patterns.to(torch.int64) & 0xFFFFFFFF
    v = torch.where(v >= 2**31, v - 2**32, v)
    return v.to(torch.int32).view(torch.uint32)


def full_range_uint32(shape, seed):
    torch.manual_seed(seed)
    patterns = torch.randint(-(2**31), 2**31, shape, dtype=torch.int64)
    return uint32_from_patterns(patterns)


def to_bits(t):
    if t.dtype == torch.uint32:
        return t.contiguous().view(torch.int32)
    if t.dtype == torch.int32:
        return t.contiguous()
    if t.dtype == torch.int64:
        return uint32_from_patterns(t).view(torch.int32)
    raise AssertionError(f"readback dtype {t.dtype} holds no 32-bit integer pattern")


def to_device(x, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(x, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


def assert_bits_equal(actual, expected, what):
    a, e = to_bits(actual), to_bits(expected)
    assert list(a.shape) == list(e.shape), f"{what}: shape {list(a.shape)} != {list(e.shape)}"
    bad = a != e
    n_bad = int(bad.sum().item())
    if n_bad:
        idx = torch.nonzero(bad.reshape(-1)).reshape(-1)[:5].tolist()
        av = a.reshape(-1).to(torch.int64) & 0xFFFFFFFF
        ev = e.reshape(-1).to(torch.int64) & 0xFFFFFFFF
        detail = ", ".join(f"[{i}] got {av[i].item():#010x} expected {ev[i].item():#010x}" for i in idx)
        raise AssertionError(f"{what}: {n_bad}/{bad.numel()} elements differ; {detail}")


def assert_untilized(out, expected):
    assert ttnn.is_tensor_storage_on_device(out), "output is not a device tensor"
    assert list(out.shape) == list(expected.shape), f"shape {list(out.shape)} != {list(expected.shape)}"
    assert out.dtype == ttnn.uint32, f"dtype {out.dtype} != uint32"
    assert out.layout == ttnn.ROW_MAJOR_LAYOUT, f"layout {out.layout} != ROW_MAJOR"
    mc = out.memory_config()
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, f"memory layout {mc.memory_layout}"
    assert mc.buffer_type == ttnn.BufferType.DRAM, f"buffer type {mc.buffer_type}"
    assert_bits_equal(ttnn.to_torch(out), expected, "output vs torch.clone")


@pytest.mark.parametrize("call_kwargs", [PRODUCTION_KWARGS, {}], ids=["model_call", "defaults"])
def test_production_case(device, call_kwargs):
    x = full_range_uint32(PRODUCTION_SHAPE, seed=42)
    out = g6_untilize(to_device(x, device), **call_kwargs)
    assert_untilized(out, torch_reference(x))


def test_face_order(device):
    # Each element holds its own row-major index, so a misplaced face or tile shows its origin.
    index = torch.arange(32 * 128, dtype=torch.int64).reshape(PRODUCTION_SHAPE)
    x = uint32_from_patterns(index + (index % 2) * 2**31)
    out = g6_untilize(to_device(x, device), **PRODUCTION_KWARGS)
    assert_untilized(out, torch_reference(x))


def test_extremes(device):
    torch.manual_seed(0)
    patterns = torch.randint(-(2**31), 2**31, PRODUCTION_SHAPE, dtype=torch.int64)
    patterns[..., 0] = 0
    patterns[..., 1] = 2**32 - 1
    patterns[..., 2] = 2**31
    patterns[..., 3] = 2**24 + 1
    x = uint32_from_patterns(patterns)
    out = g6_untilize(to_device(x, device), **PRODUCTION_KWARGS)
    assert_untilized(out, torch_reference(x))


def test_new_storage_and_input_unchanged(device):
    x = full_range_uint32(PRODUCTION_SHAPE, seed=1)
    x_dev = to_device(x, device)
    out = g6_untilize(x_dev)
    assert out.buffer_address() != x_dev.buffer_address(), "output shares the input's buffer"
    assert_untilized(out, torch_reference(x))
    assert_bits_equal(ttnn.to_torch(x_dev), x, "input after the call")


def test_repeat_is_bit_identical_and_cached(device):
    x = full_range_uint32(PRODUCTION_SHAPE, seed=2)
    x_dev = to_device(x, device)
    first = g6_untilize(x_dev, **PRODUCTION_KWARGS)
    entries = device.num_program_cache_entries()
    second = g6_untilize(x_dev, **PRODUCTION_KWARGS)
    assert device.num_program_cache_entries() == entries, "repeat call built a new program"
    # `first` stays alive, so `second` lands in a different buffer: a stale cached address shows here.
    assert second.buffer_address() != first.buffer_address()
    assert_untilized(second, torch_reference(x))
    assert_bits_equal(ttnn.to_torch(second), ttnn.to_torch(first), "repeat call")


def test_rejects_keyword_values(device, expect_error):
    x_dev = to_device(full_range_uint32(PRODUCTION_SHAPE, seed=3), device)
    with expect_error(ValueError, "use_multicore"):
        g6_untilize(x_dev, use_multicore=False)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    with expect_error(ValueError, "sub_core_grids"):
        g6_untilize(x_dev, sub_core_grids=grid)


def test_rejects_host_tensor(expect_error):
    x_host = ttnn.from_torch(full_range_uint32(PRODUCTION_SHAPE, seed=4), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)
    with expect_error((ValueError, RuntimeError), "(?i)device"):
        g6_untilize(x_host)


def test_rejects_rank_other_than_4(device, expect_error):
    x_dev = to_device(full_range_uint32((1, 32, 128), seed=5), device)
    with expect_error((ValueError, RuntimeError), "(?i)rank"):
        g6_untilize(x_dev)


def test_rejects_l1_interleaved(device, expect_error):
    x_dev = to_device(full_range_uint32(PRODUCTION_SHAPE, seed=6), device, memory_config=ttnn.L1_MEMORY_CONFIG)
    with expect_error((ValueError, RuntimeError), "(?i)dram interleaved"):
        g6_untilize(x_dev)


def test_rejects_sharded(device, expect_error):
    sharded = ttnn.create_sharded_memory_config(
        (32, 128),
        ttnn.CoreGrid(y=1, x=1),
        ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    x_dev = to_device(full_range_uint32(PRODUCTION_SHAPE, seed=7), device, memory_config=sharded)
    with expect_error((ValueError, RuntimeError), "(?i)dram interleaved"):
        g6_untilize(x_dev)
