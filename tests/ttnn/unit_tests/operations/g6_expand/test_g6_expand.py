# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_expand (immutable spec for the implementer).

out = x.expand(*sizes).contiguous(); production: x (1, 1, 1, W) -> (1, 1, W, W),
W in {512, 256, 207, 196, 185}, bfloat16, TILE, DRAM interleaved.

Gate: bit-exact equality of every logical element (helpers.TOLERANCES for
bfloat16 is ("exact", None)) -- NaN payloads, signed zeros and denormals
included. Cases are limited to the production widths and the cases the spec's
rules require (BB-22): tuple sizes, -1 sizes, special bit patterns, poisoned
input padding, repeat determinism, all widths in one process, validation.
"""

import pytest
import torch

import ttnn
from ttnn.operations.g6_expand import g6_expand

PRODUCTION_WIDTHS = [512, 256, 207, 196, 185]

# bfloat16 special bit patterns: NaNs with payloads (both signs), +-inf, -0.0,
# denormals, largest finite values.
SPECIAL_BF16_BITS = [0x7FC0, 0x7FA1, 0xFFC5, 0x7F81, 0x7F80, 0xFF80, 0x8000, 0x0001, 0x0040, 0x7F7F, 0xFF7F]


def torch_expand(x, sizes):
    """PyTorch reference: a materialized expand, same dtype."""
    return x.expand(*[int(s) for s in sizes]).contiguous()


def _bits(t):
    return t.contiguous().view(torch.int16)


def _make_input(width, *, special_values=False):
    torch.manual_seed(42)
    x = torch.randn((1, 1, 1, width), dtype=torch.float32).to(torch.bfloat16)
    if special_values:
        n = len(SPECIAL_BF16_BITS)
        row = torch.tensor([v - (1 << 16) if v >= 1 << 15 else v for v in SPECIAL_BF16_BITS], dtype=torch.int16).view(
            torch.bfloat16
        )
        x[..., :n] = row
        x[..., width - n :] = row
    return x


def _to_device(x, device):
    return ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _check(out, expected, input_tensor):
    assert list(out.shape) == list(expected.shape)
    assert out.dtype == ttnn.bfloat16
    assert out.layout == ttnn.TILE_LAYOUT
    assert out.memory_config() == input_tensor.memory_config()
    out_torch = ttnn.to_torch(out)
    assert list(out_torch.shape) == list(expected.shape)
    diff = _bits(out_torch) != _bits(expected)
    assert not bool(
        diff.any()
    ), f"{int(diff.sum())} of {diff.numel()} elements differ in bits; first at {[int(i) for i in diff.nonzero()[0]]}"
    return out_torch


@pytest.mark.parametrize("width", PRODUCTION_WIDTHS)
def test_g6_expand_production(device, width):
    x = _make_input(width)
    sizes = [1, 1, width, width]
    tt_x = _to_device(x, device)
    out = g6_expand(tt_x, sizes)
    _check(out, torch_expand(x, sizes), tt_x)


def test_g6_expand_sizes_tuple(device):
    x = _make_input(256)
    sizes = (1, 1, 256, 256)
    tt_x = _to_device(x, device)
    out = g6_expand(tt_x, sizes)
    _check(out, torch_expand(x, sizes), tt_x)


def test_g6_expand_minus_one_keeps_dim(device):
    x = _make_input(196)
    sizes = [-1, -1, 196, -1]
    tt_x = _to_device(x, device)
    out = g6_expand(tt_x, sizes)
    _check(out, torch_expand(x, sizes), tt_x)


def test_g6_expand_special_values(device):
    x = _make_input(196, special_values=True)
    sizes = [1, 1, 196, 196]
    tt_x = _to_device(x, device)
    assert torch.equal(_bits(ttnn.to_torch(tt_x)), _bits(x)), "harness did not carry the bit patterns"
    out = g6_expand(tt_x, sizes)
    _check(out, torch_expand(x, sizes), tt_x)


@pytest.mark.parametrize("width", [512, 196])
def test_g6_expand_poisoned_input_padding(device, width):
    """Rows 1..31 of every x tile and columns beyond W hold 1000.0; out must not see them."""
    x = _make_input(width)
    sizes = [1, 1, width, width]
    tt_x = ttnn.fill_implicit_tile_padding(_to_device(x, device), 1000.0)
    out = g6_expand(tt_x, sizes)
    _check(out, torch_expand(x, sizes), tt_x)


def test_g6_expand_repeat_bit_identical(device):
    x = _make_input(512)
    sizes = [1, 1, 512, 512]
    tt_x = _to_device(x, device)
    first = _check(g6_expand(tt_x, sizes), torch_expand(x, sizes), tt_x)
    second = _check(g6_expand(tt_x, sizes), torch_expand(x, sizes), tt_x)
    assert torch.equal(_bits(first), _bits(second))


def test_g6_expand_all_widths_one_process(device):
    """Program cache across widths: each call graded against its own reference."""
    for width in [512, 196, 512, 185, 207, 256]:
        x = _make_input(width)
        sizes = [1, 1, width, width]
        tt_x = _to_device(x, device)
        _check(g6_expand(tt_x, sizes), torch_expand(x, sizes), tt_x)


@pytest.mark.parametrize(
    "shape, sizes",
    [
        ((1, 1, 1, 64), [1, 64, 64]),  # sizes length differs from rank
        ((1, 1, 1, 64), [1, 1, 0, 64]),  # size 0
        ((1, 1, 1, 64), [1, 1, -2, 64]),  # negative size other than -1
        ((1, 1, 1, 64), [1, 1, 64, 32]),  # differs along a dim whose size is not 1
        ((1, 1, 1, 64), [2, 1, 64, 64]),  # expansion of dim 0 (outside the contract)
        ((1, 1, 1, 1), [1, 1, 32, 64]),  # expansion of dim 3 (outside the contract)
    ],
)
def test_g6_expand_invalid_sizes(device, expect_error, shape, sizes):
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    tt_x = _to_device(x, device)
    with expect_error((ValueError, RuntimeError), ""):
        g6_expand(tt_x, sizes)


def test_g6_expand_rank_not_4(device, expect_error):
    x = torch.randn((1, 1, 64), dtype=torch.float32).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error((ValueError, RuntimeError), ""):
        g6_expand(tt_x, [1, 64, 64])


def test_g6_expand_input_not_on_device(expect_error):
    x = torch.randn((1, 1, 1, 64), dtype=torch.float32).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error((ValueError, RuntimeError), ""):
        g6_expand(tt_x, [1, 1, 64, 64])


def test_g6_expand_input_in_l1(device, expect_error):
    x = torch.randn((1, 1, 1, 64), dtype=torch.float32).to(torch.bfloat16)
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    with expect_error((ValueError, RuntimeError), ""):
        g6_expand(tt_x, [1, 1, 64, 64])
