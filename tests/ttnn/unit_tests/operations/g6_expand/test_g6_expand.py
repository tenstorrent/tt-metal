# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_expand (immutable spec for the implementer).

Bit-exact gate: helpers.TOLERANCES[bfloat16] is ("exact", None), so every logical
element of out must equal torch.expand(*sizes).contiguous() bit for bit.
Limited to the production cases and the cases the spec's rules require (BB-22).
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_expand import g6_expand

PRODUCTION_WIDTHS = [512, 256, 196, 185, 207]

# bfloat16 special bit patterns: NaNs with payloads (both signs), +-inf, -0.0,
# denormals, largest finite values.
SPECIAL_BF16_BITS = [0x7FC0, 0x7FA1, 0xFFC5, 0x7F81, 0x7F80, 0xFF80, 0x8000, 0x0001, 0x0040, 0x7F7F, 0xFF7F]


def torch_reference(x, sizes):
    return x.expand(*[int(s) for s in sizes]).contiguous()


def _bits(t):
    return t.contiguous().view(torch.int16)


def _to_device(x, device):
    return ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _check(out, expected):
    assert list(out.shape) == list(expected.shape)
    assert out.dtype == ttnn.bfloat16
    assert out.layout == ttnn.TILE_LAYOUT
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert out.memory_config().buffer_type == ttnn.BufferType.DRAM
    got = ttnn.to_torch(out)
    assert got.dtype == torch.bfloat16
    diff = _bits(got) != _bits(expected)
    assert not bool(diff.any()), f"{int(diff.sum())} of {diff.numel()} elements differ in bits"
    return got


def _special_row(width):
    torch.manual_seed(42)
    x = torch.randn((1, 1, 1, width), dtype=torch.float32).to(torch.bfloat16)
    row = torch.tensor([v - (1 << 16) if v >= 1 << 15 else v for v in SPECIAL_BF16_BITS], dtype=torch.int16).view(
        torch.bfloat16
    )
    n = len(SPECIAL_BF16_BITS)
    x[..., :n] = row
    x[..., width - n :] = row
    return x


@pytest.mark.parametrize("width", PRODUCTION_WIDTHS)
def test_g6_expand_production(device, width):
    torch.manual_seed(42)
    x = torch.randn((1, 1, 1, width), dtype=torch.float32).to(torch.bfloat16)
    sizes = [1, 1, width, width]
    out = g6_expand(_to_device(x, device), sizes)
    _check(out, torch_reference(x, sizes))


@pytest.mark.parametrize("width", [196, 512])
def test_g6_expand_special_values(device, width):
    x = _special_row(width)
    tt_x = _to_device(x, device)
    assert torch.equal(_bits(ttnn.to_torch(tt_x)), _bits(x)), "harness did not carry special bits"
    sizes = [1, 1, width, width]
    _check(g6_expand(tt_x, sizes), torch_reference(x, sizes))


@pytest.mark.parametrize("width", [196, 512])
def test_g6_expand_poisoned_input_padding(device, width):
    """Rows 1..31 of every tile of x and its columns beyond W hold 1000.0;
    no logical element of out may depend on them."""
    torch.manual_seed(42)
    x = torch.randn((1, 1, 1, width), dtype=torch.float32).to(torch.bfloat16)
    tt_x = ttnn.fill_implicit_tile_padding(_to_device(x, device), 1000.0)
    sizes = [1, 1, width, width]
    _check(g6_expand(tt_x, sizes), torch_reference(x, sizes))


def test_g6_expand_sizes_tuple_and_minus_one(device):
    torch.manual_seed(42)
    x = torch.randn((1, 1, 1, 256), dtype=torch.float32).to(torch.bfloat16)
    tt_x = _to_device(x, device)
    _check(g6_expand(tt_x, (1, 1, 256, 256)), torch_reference(x, (1, 1, 256, 256)))
    _check(g6_expand(tt_x, [-1, -1, 256, -1]), torch_reference(x, [1, 1, 256, 256]))


def test_g6_expand_new_tensor_and_repeat(device):
    """Result owns its storage; repeated calls (program cache) are bit-identical,
    and calls of different widths in one process each match their own oracle."""
    torch.manual_seed(42)
    xs = {w: torch.randn((1, 1, 1, w), dtype=torch.float32).to(torch.bfloat16) for w in (512, 196)}
    tt_xs = {w: _to_device(x, device) for w, x in xs.items()}
    first = None
    for w in (512, 196, 512, 512):
        sizes = [1, 1, w, w]
        out = g6_expand(tt_xs[w], sizes)
        got = _check(out, torch_reference(xs[w], sizes))
        if w == 512:
            if first is None:
                first = got
            else:
                assert torch.equal(_bits(got), _bits(first))
    # Input untouched.
    assert torch.equal(_bits(ttnn.to_torch(tt_xs[512])), _bits(xs[512]))


@pytest.mark.parametrize(
    "sizes",
    [
        [1, 256, 256],  # wrong length
        [1, 1, 0, 256],  # size 0
        [1, 1, -2, 256],  # negative other than -1
        [1, 1, 256, 128],  # differs from a non-1 dim
        [2, 1, 256, 256],  # expands dim 0 (outside contract)
    ],
)
def test_g6_expand_rejects_bad_sizes(device, sizes, expect_error):
    torch.manual_seed(42)
    x = torch.randn((1, 1, 1, 256), dtype=torch.float32).to(torch.bfloat16)
    with expect_error((ValueError, RuntimeError), ""):
        g6_expand(_to_device(x, device), sizes)


def test_g6_expand_rejects_l1_input(device, expect_error):
    torch.manual_seed(42)
    x = torch.randn((1, 1, 1, 256), dtype=torch.float32).to(torch.bfloat16)
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    with expect_error((ValueError, RuntimeError), ""):
        g6_expand(tt_x, [1, 1, 256, 256])


def test_g6_expand_rejects_host_input(expect_error):
    x = torch.randn((1, 1, 1, 256), dtype=torch.float32).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error((ValueError, RuntimeError), ""):
        g6_expand(tt_x, [1, 1, 256, 256])
