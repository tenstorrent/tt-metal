# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_untilize: uint32 TILE -> ROW_MAJOR, bit-identical.

Shapes are limited to the one production case, (1, 1, 32, 128) uint32 TILE in
DRAM (BB-22), plus the cases the spec's rules require: validation refusals,
new storage, program-cache reuse with moving addresses, repeat determinism.
Values are compared on 32-bit patterns (helpers.TOLERANCES: ("exact", None));
inputs are drawn uniformly over all 2^32 patterns, never a float distribution,
because small integers survive a float32 or 16-bit path.
"""

import pytest
import torch
import ttnn

from ttnn.operations._op_contract import SupportRefusal
from ttnn.operations.g6_untilize import g6_untilize

SHAPE = (1, 1, 32, 128)
PRODUCTION_KWARGS = {"use_multicore": True, "sub_core_grids": None}
# Every refusal names the op (op_design.md, Host contract).
REFUSAL_MESSAGE = "g6_untilize"


def pytorch_reference(input_tensor):
    return input_tensor.clone()


def uint32_from_patterns(patterns):
    """int64 values taken mod 2^32 -> a torch.uint32 tensor with those bit patterns."""
    v = patterns.to(torch.int64) & 0xFFFFFFFF
    v = torch.where(v >= 2**31, v - 2**32, v)
    return v.to(torch.int32).view(torch.uint32)


def random_uint32(shape):
    patterns = torch.randint(-(2**31), 2**31, shape, dtype=torch.int64)
    return uint32_from_patterns(patterns)


def bits32(t):
    if t.dtype == torch.uint32:
        return t.contiguous().view(torch.int32)
    if t.dtype == torch.int32:
        return t
    if t.dtype == torch.int64:
        return uint32_from_patterns(t).view(torch.int32)
    raise AssertionError(f"readback dtype {t.dtype} holds no 32-bit integer pattern")


def to_device(torch_input, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        torch_input, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )


def assert_bits_equal(actual, expected, what):
    assert list(actual.shape) == list(expected.shape), f"{what}: shape {list(actual.shape)} != {list(expected.shape)}"
    a, e = bits32(actual), bits32(expected)
    bad = a != e
    n_bad = int(bad.sum().item())
    if n_bad:
        idx = torch.nonzero(bad.reshape(-1)).reshape(-1)[:5].tolist()
        av = a.reshape(-1).to(torch.int64) & 0xFFFFFFFF
        ev = e.reshape(-1).to(torch.int64) & 0xFFFFFFFF
        detail = ", ".join(f"[{i}] got {av[i].item():#010x} expected {ev[i].item():#010x}" for i in idx)
        raise AssertionError(f"{what}: {n_bad}/{bad.numel()} elements differ; {detail}")


def check_output(ttnn_output, torch_input):
    """Output contract (device, shape, dtype, ROW_MAJOR, DRAM interleaved) and bit-exact values."""
    assert ttnn.is_tensor_storage_on_device(ttnn_output), "output is not a device tensor"
    assert list(ttnn_output.shape) == list(torch_input.shape)
    assert ttnn_output.dtype == ttnn.uint32, f"dtype {ttnn_output.dtype}"
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT, f"layout {ttnn_output.layout}"
    mc = ttnn_output.memory_config()
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, f"memory_config {mc}"
    assert mc.buffer_type == ttnn.BufferType.DRAM, f"memory_config {mc}"
    actual = ttnn.to_torch(ttnn_output)
    assert_bits_equal(actual, pytorch_reference(torch_input), "output vs torch.clone")
    return actual


def test_g6_untilize_random_full_range(device):
    """Production call on uniform full-range uint32 patterns; new storage; input unchanged."""
    torch.manual_seed(42)
    torch_input = random_uint32(SHAPE)
    ttnn_input = to_device(torch_input, device)

    ttnn_output = g6_untilize(ttnn_input, **PRODUCTION_KWARGS)

    check_output(ttnn_output, torch_input)
    assert ttnn_output.buffer_address() != ttnn_input.buffer_address(), "output shares the input's buffer"
    assert ttnn_input.layout == ttnn.TILE_LAYOUT
    assert_bits_equal(ttnn.to_torch(ttnn_input), torch_input, "input after the call")


def test_g6_untilize_defaults_call(device):
    """g6_untilize(x): the model's call with its defaults."""
    torch.manual_seed(42)
    torch_input = random_uint32(SHAPE)
    check_output(g6_untilize(to_device(torch_input, device)), torch_input)


def test_g6_untilize_face_order(device):
    """Element (r, w) holds r * 128 + w, plus 2^31 on odd elements: a wrong face, tile
    or row order names the misplaced element, and the high bit exposes a narrow path."""
    index = torch.arange(SHAPE[2] * SHAPE[3], dtype=torch.int64)
    torch_input = uint32_from_patterns(index + (index % 2) * 2**31).reshape(SHAPE)
    check_output(g6_untilize(to_device(torch_input, device), **PRODUCTION_KWARGS), torch_input)


def test_g6_untilize_extremes(device):
    """0, 2^32 - 1, 2^31, 2^24 + 1 in every row and in both faces of every tile."""
    torch.manual_seed(42)
    patterns = torch.randint(-(2**31), 2**31, SHAPE, dtype=torch.int64)
    extremes = torch.tensor([0, 2**32 - 1, 2**31, 2**24 + 1], dtype=torch.int64)
    for col in (0, 16, 32, 48, 64, 80, 96, 112):
        patterns[..., col : col + 4] = extremes
    torch_input = uint32_from_patterns(patterns)
    check_output(g6_untilize(to_device(torch_input, device), **PRODUCTION_KWARGS), torch_input)


def test_g6_untilize_program_cache_and_repeat(device):
    """Repeat calls reuse one program while the buffers move: fresh data at a new address
    each call must come out right (a stale address shows as a mismatch), and the same
    input twice must give bit-identical outputs (BB-06, BB-07)."""
    torch.manual_seed(42)
    spacers, input_addresses = [], set()
    entries_after_first = None
    for i in range(3):
        spacers.append(ttnn.from_torch(torch.zeros((32, 32 * (i + 1))), layout=ttnn.TILE_LAYOUT, device=device))
        torch_input = random_uint32(SHAPE)
        ttnn_input = to_device(torch_input, device)
        input_addresses.add(ttnn_input.buffer_address())

        first = check_output(g6_untilize(ttnn_input, **PRODUCTION_KWARGS), torch_input)
        second = ttnn.to_torch(g6_untilize(ttnn_input, **PRODUCTION_KWARGS))
        assert_bits_equal(second, first, f"repeat call {i}")

        if entries_after_first is None:
            entries_after_first = device.num_program_cache_entries()

    assert len(input_addresses) > 1, "the test did not move the input buffer"
    assert device.num_program_cache_entries() == entries_after_first, "a repeat call built a new program"


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"use_multicore": False, "sub_core_grids": None}, id="use_multicore_false"),
        pytest.param(
            {
                "use_multicore": True,
                "sub_core_grids": ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
            },
            id="sub_core_grids_set",
        ),
    ],
)
def test_g6_untilize_refuses_keywords(device, expect_error, kwargs):
    torch.manual_seed(42)
    ttnn_input = to_device(random_uint32(SHAPE), device)
    with expect_error(ValueError, REFUSAL_MESSAGE):
        g6_untilize(ttnn_input, **kwargs)


def _height_sharded_l1():
    return ttnn.create_sharded_memory_config(
        shape=(SHAPE[2], SHAPE[3]),
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )


@pytest.mark.parametrize("case", ["host_tensor", "rank_3", "l1_interleaved", "height_sharded_l1"])
def test_g6_untilize_refuses_inputs(device, expect_error, case):
    torch.manual_seed(42)
    if case == "host_tensor":
        ttnn_input = ttnn.from_torch(random_uint32(SHAPE), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)
    elif case == "rank_3":
        ttnn_input = to_device(random_uint32(SHAPE[1:]), device)
    elif case == "l1_interleaved":
        ttnn_input = to_device(random_uint32(SHAPE), device, memory_config=ttnn.L1_MEMORY_CONFIG)
    else:
        ttnn_input = to_device(random_uint32(SHAPE), device, memory_config=_height_sharded_l1())
    with expect_error((ValueError, RuntimeError), REFUSAL_MESSAGE):
        g6_untilize(ttnn_input, **PRODUCTION_KWARGS)


@pytest.mark.parametrize("case", ["float32_tile", "uint32_row_major"])
def test_g6_untilize_refuses_unsupported_cells(device, expect_error, case):
    """Cells outside SUPPORTED raise a support refusal from validate()."""
    torch.manual_seed(42)
    if case == "float32_tile":
        ttnn_input = ttnn.from_torch(
            torch.randn(SHAPE, dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        ttnn_input = ttnn.from_torch(
            random_uint32(SHAPE),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    with expect_error(SupportRefusal, REFUSAL_MESSAGE):
        g6_untilize(ttnn_input, **PRODUCTION_KWARGS)
