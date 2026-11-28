# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_unsqueeze.

Scope is set by rule BB-22 of the op spec: the production cases only (float32
TILE DRAM-interleaved [1, 512] and [1, 256], new axis last as dim 2 and as -1)
plus the cases the spec's rules name: special bit patterns, NaN in the input
tile padding, a repeated call, program reuse across widths, and the host-side
refusals. Other shapes are supported but deliberately not swept.

The op moves values and never converts them, so every comparison is on int32
views: NaN payloads, -0.0, infinities and subnormals must survive bit for bit,
and torch.equal on floats cannot grade NaN.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_unsqueeze import g6_unsqueeze

PRODUCTION_CASES = [
    pytest.param((1, 512), 2, id="1x512-dim2"),
    pytest.param((1, 256), 2, id="1x256-dim2"),
    pytest.param((1, 512), -1, id="1x512-dim-1"),
]

# Quiet NaN, quiet NaN with the sign bit, signaling NaN: a path that
# canonicalizes or quiets NaN changes these payloads.
_NAN_BITS = (0x7FC12345, 0xFFFC5433, 0x7F800001)
_F32 = torch.finfo(torch.float32)
# Subnormals are zeroed by a flushing path; the extremes overflow a narrower one.
_SPECIAL_VALUES = (float("inf"), float("-inf"), -0.0, 1e-40, -1e-45, _F32.max, -_F32.max, _F32.tiny, -_F32.tiny)
# 3 is coprime to 32, so the special patterns reach every lane of both tile halves.
_SPECIAL_STRIDE = 3


def torch_reference(x, dim):
    return torch.unsqueeze(x, dim)


def _bits(t):
    return t.contiguous().view(torch.int32)


def _to_device(x, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


def _special_values_input(shape):
    x = torch.randn(shape, dtype=torch.float32)
    nan_bits = torch.tensor([b - 2**32 if b >= 2**31 else b for b in _NAN_BITS], dtype=torch.int32)
    value_bits = torch.tensor(_SPECIAL_VALUES, dtype=torch.float32).view(torch.int32)
    pool = torch.cat([nan_bits, value_bits])
    flat_bits = x.view(-1).view(torch.int32)
    positions = torch.arange(0, flat_bits.numel(), _SPECIAL_STRIDE)
    flat_bits[positions] = pool[torch.arange(positions.numel()) % pool.numel()]
    return x


def _require_round_trip(tt_input, x):
    """Skip when from_torch/to_torch itself changes input bits: the op cannot be graded through that."""
    back = ttnn.to_torch(tt_input)
    if list(back.shape) != list(x.shape) or not torch.equal(_bits(back), _bits(x)):
        pytest.skip("harness: the host round trip changed input bit patterns; bit exactness cannot be graded")


def _assert_output(tt_output, tt_input, expected):
    assert tt_output.dtype == ttnn.float32
    assert tt_output.layout == ttnn.TILE_LAYOUT
    in_mc, out_mc = tt_input.memory_config(), tt_output.memory_config()
    assert out_mc.buffer_type == in_mc.buffer_type == ttnn.BufferType.DRAM
    assert out_mc.memory_layout == in_mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert tt_output.buffer_address() != tt_input.buffer_address(), "the output must be a tensor of its own"
    actual = ttnn.to_torch(tt_output)
    assert list(actual.shape) == list(expected.shape)
    wrong = (_bits(actual) != _bits(expected)).reshape(-1)
    if bool(wrong.any()):
        first = int(wrong.nonzero()[0])
        got = int(_bits(actual).reshape(-1)[first]) & 0xFFFFFFFF
        want = int(_bits(expected).reshape(-1)[first]) & 0xFFFFFFFF
        pytest.fail(
            f"{int(wrong.sum())} of {wrong.numel()} values differ bitwise; "
            f"first at flat index {first}: got 0x{got:08X}, expected 0x{want:08X}"
        )


@pytest.mark.parametrize("shape,dim", PRODUCTION_CASES)
def test_production_bit_exact(device, shape, dim):
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.float32)
    tt_input = _to_device(x, device)
    _assert_output(g6_unsqueeze(tt_input, dim), tt_input, torch_reference(x, dim))


def test_dim_keyword(device):
    torch.manual_seed(42)
    x = torch.randn((1, 512), dtype=torch.float32)
    tt_input = _to_device(x, device)
    _assert_output(g6_unsqueeze(tt_input, dim=2), tt_input, torch_reference(x, 2))


def test_special_values_bit_exact(device):
    torch.manual_seed(42)
    x = _special_values_input((1, 512))
    tt_input = _to_device(x, device)
    _require_round_trip(tt_input, x)
    _assert_output(g6_unsqueeze(tt_input, 2), tt_input, torch_reference(x, 2))


def test_poisoned_padding_never_reaches_output(device):
    torch.manual_seed(42)
    x = torch.randn((1, 256), dtype=torch.float32)
    tt_input = ttnn.fill_implicit_tile_padding(_to_device(x, device), float("nan"))
    padded = tt_input.cpu().to_torch_with_padded_shape()
    if not bool(torch.isnan(padded[..., 1:, :]).all()):
        pytest.skip("harness: NaN did not reach the input tile padding")
    _assert_output(g6_unsqueeze(tt_input, 2), tt_input, torch_reference(x, 2))


def test_repeat_call_bit_identical(device):
    torch.manual_seed(42)
    x = torch.randn((1, 512), dtype=torch.float32)
    tt_input = _to_device(x, device)
    first = ttnn.to_torch(g6_unsqueeze(tt_input, 2))
    second = ttnn.to_torch(g6_unsqueeze(tt_input, 2))
    assert torch.equal(_bits(first), _bits(torch_reference(x, 2)))
    assert torch.equal(_bits(second), _bits(first))


def test_program_reused_per_width(device):
    """p2 then p1 build one program each; a repeat of p2 builds none."""
    device.clear_program_cache()
    device.enable_program_cache()
    torch.manual_seed(42)
    for shape in [(1, 256), (1, 512)]:
        x = torch.randn(shape, dtype=torch.float32)
        tt_input = _to_device(x, device)
        _assert_output(g6_unsqueeze(tt_input, 2), tt_input, torch_reference(x, 2))
    built = device.num_program_cache_entries()
    x = torch.randn((1, 256), dtype=torch.float32)
    tt_input = _to_device(x, device)
    _assert_output(g6_unsqueeze(tt_input, 2), tt_input, torch_reference(x, 2))
    assert device.num_program_cache_entries() == built


_REFUSED = [
    pytest.param((1, 512), 0, "dram", id="1x512-dim0"),
    pytest.param((1, 512), 1, "dram", id="1x512-dim1"),
    pytest.param((1, 512), -2, "dram", id="1x512-dim-2"),
    pytest.param((1, 512), -3, "dram", id="1x512-dim-3"),
    pytest.param((1, 512), 3, "dram", id="1x512-dim3-out_of_range"),
    pytest.param((1, 512), -4, "dram", id="1x512-dim-4-out_of_range"),
    pytest.param((1, 1, 1, 32), 4, "dram", id="1x1x1x32-dim4-rank4"),
    pytest.param((2, 64), 2, "dram", id="2x64-dim2-two_rows"),
    pytest.param((1, 512), 2, "l1", id="1x512-dim2-l1_interleaved"),
    pytest.param((1, 0), 2, "dram", id="1x0-dim2-empty"),
]


@pytest.mark.parametrize("shape,dim,buffer", _REFUSED)
def test_malformed_request_refused_on_host(device, expect_error, shape, dim, buffer):
    memory_config = ttnn.L1_MEMORY_CONFIG if buffer == "l1" else ttnn.DRAM_MEMORY_CONFIG
    try:
        tt_input = _to_device(torch.randn(shape, dtype=torch.float32), device, memory_config)
    except Exception as exc:  # noqa: BLE001 — a construction failure belongs to the harness, not the op
        pytest.skip(f"harness: cannot build a {list(shape)} {buffer} TILE input: {exc!r}")
    built = device.num_program_cache_entries()
    with expect_error((ValueError, RuntimeError), "g6_unsqueeze"):
        g6_unsqueeze(tt_input, dim)
    assert device.num_program_cache_entries() == built, "the refusal came after dispatch"
