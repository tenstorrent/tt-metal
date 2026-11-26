# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_expand: x (1, 1, 1, W) -> (1, 1, W, W), every logical element bit-exact.

Limited (BB-22) to the five production widths and the cases the spec's rules require:
sizes as a tuple and with -1, special bfloat16 bit patterns, poisoned input tile padding,
a new output tensor in the input's placement, repeat determinism, the program cache across
widths, and the validation errors. bfloat16, TILE, DRAM interleaved.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_expand import g6_expand

PRODUCTION_WIDTHS = [512, 256, 196, 185, 207]

# Canonical NaN, NaN payloads (quiet and signalling, both signs), +-inf, -0.0,
# smallest and a mid denormal, largest finite values.
SPECIAL_BF16_BITS = [0x7FC0, 0x7FA1, 0xFFC5, 0x7F81, 0x7F80, 0xFF80, 0x8000, 0x0001, 0x0040, 0x7F7F, 0xFF7F]


def torch_g6_expand(x, sizes):
    return x.expand(*[int(s) for s in sizes]).contiguous()


def _bits(t):
    # Bit patterns, so NaN payloads and -0.0 compare exactly.
    return t.contiguous().view(torch.int16)


def _random_row(width):
    return torch.randn((1, 1, 1, width), dtype=torch.float32).to(torch.bfloat16)


def _special_row(width):
    x = _random_row(width)
    n = len(SPECIAL_BF16_BITS)
    row = torch.tensor([v - (1 << 16) if v >= 1 << 15 else v for v in SPECIAL_BF16_BITS], dtype=torch.int16).view(
        torch.bfloat16
    )
    x[..., :n] = row
    x[..., width - n :] = row
    return x


def _to_device(x, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


def _assert_bit_exact(got, expected):
    assert list(got.shape) == list(expected.shape), f"shape {list(got.shape)} != {list(expected.shape)}"
    assert got.dtype == expected.dtype
    diff = _bits(got) != _bits(expected)
    if bool(diff.any()):
        first = diff.nonzero()[0].tolist()
        pytest.fail(f"{int(diff.sum())} of {diff.numel()} elements differ in bits; first at {first}")


def _check_output(out, expected, x_tt):
    assert list(out.shape) == list(expected.shape)
    assert out.dtype == ttnn.bfloat16
    assert out.layout == ttnn.TILE_LAYOUT
    memory_config = out.memory_config()
    assert memory_config.buffer_type == ttnn.BufferType.DRAM
    assert memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert out.buffer_address() != x_tt.buffer_address(), "out must own its storage, not alias the input"
    got = ttnn.to_torch(out)
    _assert_bit_exact(got, expected)
    return got


def _run(x, sizes, device, x_tt=None):
    expected = torch_g6_expand(x, sizes)
    if x_tt is None:
        x_tt = _to_device(x, device)
    out = g6_expand(x_tt, sizes)
    _check_output(out, expected, x_tt)
    return x_tt, out


@pytest.mark.parametrize("width", PRODUCTION_WIDTHS, ids=[f"W{w}" for w in PRODUCTION_WIDTHS])
def test_production_cases(width, device):
    torch.manual_seed(42)
    x = _random_row(width)
    x_tt, _ = _run(x, [1, 1, width, width], device)
    _assert_bit_exact(ttnn.to_torch(x_tt), x)  # the input is left untouched


def test_sizes_as_tuple(device):
    torch.manual_seed(42)
    x = _random_row(256)
    _run(x, (1, 1, 256, 256), device)


def test_sizes_with_minus_one(device):
    torch.manual_seed(42)
    x = _random_row(512)
    _run(x, [-1, -1, 512, -1], device)


@pytest.mark.parametrize("width", [196, 512], ids=["W196", "W512"])
def test_special_values_bit_identical(width, device):
    torch.manual_seed(42)
    x = _special_row(width)
    x_tt = _to_device(x, device)
    if not torch.equal(_bits(ttnn.to_torch(x_tt)), _bits(x)):
        pytest.fail("harness: ttnn.from_torch did not carry the bfloat16 special bit patterns unchanged")
    _run(x, [1, 1, width, width], device, x_tt=x_tt)


@pytest.mark.parametrize("width", [512, 196], ids=["W512", "W196"])
def test_poisoned_input_padding(width, device):
    # Rows 1..31 of every input tile and the columns beyond W hold 1000.0; no logical
    # element of out may depend on them.
    torch.manual_seed(42)
    x = _random_row(width)
    x_tt = ttnn.fill_implicit_tile_padding(_to_device(x, device), 1000.0)
    _run(x, [1, 1, width, width], device, x_tt=x_tt)


def test_repeat_is_bit_identical_and_cached(device):
    torch.manual_seed(42)
    x = _random_row(512)
    sizes = [1, 1, 512, 512]
    expected = torch_g6_expand(x, sizes)
    x_tt = _to_device(x, device)
    first = g6_expand(x_tt, sizes)
    entries = device.num_program_cache_entries()
    second = g6_expand(x_tt, sizes)
    assert device.num_program_cache_entries() == entries, "a repeat call with the same shapes built a new program"
    first_host = _check_output(first, expected, x_tt)
    second_host = _check_output(second, expected, x_tt)
    _assert_bit_exact(second_host, first_host)


def test_width_sequence_under_program_cache(device):
    # Every production width in one process, each graded against its own oracle; fresh data per
    # call so a replayed program with a stale buffer address cannot pass.
    torch.manual_seed(42)
    for width in [512, 196, 512, 185, 207, 256]:
        x = _random_row(width)
        _run(x, [1, 1, width, width], device)


def _host_row(width):
    return ttnn.from_torch(_random_row(width), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


VALIDATION_CASES = {
    "sizes_length_3": (lambda d: _to_device(_random_row(512), d), [1, 512, 512]),
    "rank_3_input": (
        lambda d: ttnn.from_torch(
            torch.randn((1, 1, 512)).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=d,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        [1, 512, 512],
    ),
    "size_zero": (lambda d: _to_device(_random_row(512), d), [1, 1, 0, 512]),
    "negative_size": (lambda d: _to_device(_random_row(512), d), [1, 1, -32, 512]),
    "size_differs_on_non_one_dim": (lambda d: _to_device(_random_row(512), d), [1, 1, 512, 256]),
    "expand_dim_0": (lambda d: _to_device(_random_row(512), d), [2, 1, 512, 512]),
    "expand_dim_1": (lambda d: _to_device(_random_row(512), d), [1, 2, 512, 512]),
    "l1_input": (lambda d: _to_device(_random_row(512), d, ttnn.L1_MEMORY_CONFIG), [1, 1, 512, 512]),
    "host_input": (lambda d: _host_row(512), [1, 1, 512, 512]),
}


@pytest.mark.parametrize("case", list(VALIDATION_CASES), ids=list(VALIDATION_CASES))
def test_validation_errors(case, device, expect_error):
    torch.manual_seed(42)
    make_input, sizes = VALIDATION_CASES[case]
    x_tt = make_input(device)
    # The spec fixes the error type, not its message.
    with expect_error((ValueError, RuntimeError), ""):
        g6_expand(x_tt, sizes)
