# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for `ttnn.operations.tilize.tilize`.

This file is the SPEC. The implementer must not modify it.

`tilize` is a pure layout conversion: ROW_MAJOR -> TILE, values unchanged. The
PyTorch reference is therefore the identity function -- the test asserts that
the round trip through TILE layout preserves the tensor.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.tilize import tilize


# PCC tolerance keyed by dtype -- identical to the golden suite's thresholds.
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}


def torch_reference(x: torch.Tensor) -> torch.Tensor:
    """tilize reorders bytes only -- the value-level reference is the identity."""
    return x


# (single-tile, multi-tile, non-square wide, non-square tall, multi-batch, tall)
SHAPES = [
    pytest.param([1, 1, 32, 32], id="single_tile"),
    pytest.param([1, 1, 64, 128], id="multi_tile"),
    pytest.param([1, 1, 32, 256], id="non_square_wide"),
    pytest.param([1, 1, 256, 32], id="non_square_tall"),
    pytest.param([2, 3, 64, 96], id="multi_batch"),
    pytest.param([1, 1, 1024, 64], id="tall"),
]

DTYPES = [
    pytest.param(ttnn.bfloat16, id="bfloat16"),
    pytest.param(ttnn.float32, id="float32"),
]


def _make_input(shape, dtype, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        torch_input = torch_input.to(torch.bfloat16).to(torch.float32)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    return torch_input, tt_input


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("use_multicore", [False, True], ids=["single_core", "multi_core"])
def test_tilize(device, shape, dtype, use_multicore):
    """Core contract: RM -> TILE preserves every value, single- and multi-core."""
    torch_input, tt_input = _make_input(shape, dtype, device)

    tt_output = tilize(tt_input, use_multicore=use_multicore)

    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert list(tt_output.shape) == list(shape)

    torch_output = ttnn.to_torch(tt_output).to(torch.float32)
    assert_with_pcc(torch_reference(torch_input), torch_output, PCC[dtype])


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_tilize_single_buffered(device, shape, dtype):
    """`use_double_buffer=False` changes only L1 footprint -- never the values."""
    torch_input, tt_input = _make_input(shape, dtype, device)

    tt_output = tilize(tt_input, use_double_buffer=False)

    torch_output = ttnn.to_torch(tt_output).to(torch.float32)
    assert_with_pcc(torch_reference(torch_input), torch_output, PCC[dtype])


@pytest.mark.parametrize(
    "memory_config",
    [
        pytest.param(ttnn.DRAM_MEMORY_CONFIG, id="dram_out"),
        pytest.param(ttnn.L1_MEMORY_CONFIG, id="l1_out"),
    ],
)
def test_tilize_memory_config(device, memory_config):
    """An explicit output memory_config is honoured and does not disturb values."""
    torch_input, tt_input = _make_input([1, 1, 128, 128], ttnn.bfloat16, device)

    tt_output = tilize(tt_input, memory_config=memory_config)

    torch_output = ttnn.to_torch(tt_output).to(torch.float32)
    assert_with_pcc(torch_reference(torch_input), torch_output, PCC[ttnn.bfloat16])


@pytest.mark.parametrize(
    "out_dtype",
    [
        pytest.param(ttnn.bfloat16, id="to_bfloat16"),
        pytest.param(ttnn.bfloat8_b, id="to_bfloat8_b"),
    ],
)
def test_tilize_dtype_cast(device, out_dtype):
    """`dtype=` performs a real value-preserving cast at pack time."""
    torch_input, tt_input = _make_input([1, 1, 128, 128], ttnn.float32, device)

    tt_output = tilize(tt_input, dtype=out_dtype)

    assert tt_output.dtype == out_dtype
    torch_output = ttnn.to_torch(tt_output).to(torch.float32)
    assert_with_pcc(torch_reference(torch_input), torch_output, PCC[out_dtype])


@pytest.mark.parametrize("shape", [[64, 128], [3, 64, 96], [1, 2, 3, 64, 32]], ids=["rank2", "rank3", "rank5"])
def test_tilize_ranks(device, shape):
    """Leading dims fold into the tile-row count rank-agnostically."""
    torch_input, tt_input = _make_input(shape, ttnn.bfloat16, device)

    tt_output = tilize(tt_input)

    assert list(tt_output.shape) == list(shape)
    torch_output = ttnn.to_torch(tt_output).to(torch.float32)
    assert_with_pcc(torch_reference(torch_input), torch_output, PCC[ttnn.bfloat16])


def test_tilize_unaligned_without_padding_raises(device, expect_error):
    """Padding is never implicit: an unaligned input with no pad args must raise.

    The refusal message MUST mention "pad" -- see op_design.md section 2
    (Canonicalization), which pins that requirement.
    """
    _, tt_input = _make_input([1, 1, 50, 50], ttnn.bfloat16, device)

    with expect_error((ValueError, RuntimeError), "(?i)pad"):
        tilize(tt_input)


@pytest.mark.parametrize(
    "shape, target, pad_value",
    [
        pytest.param([1, 1, 50, 50], [1, 1, 64, 64], 10.0, id="tile_rounded_positive"),
        pytest.param([1, 1, 50, 50], [1, 1, 128, 128], -18.0, id="beyond_tile_round_negative"),
        pytest.param([1, 1, 32, 50], [1, 1, 32, 128], 3.5, id="w_only_beyond_tile_round"),
    ],
)
def test_tilize_explicit_padding(device, shape, target, pad_value):
    """Padding changes the PADDED shape, never the LOGICAL shape; both views hold."""
    torch_input, tt_input = _make_input(shape, ttnn.bfloat16, device)

    tt_output = tilize(tt_input, output_padded_shape=target, pad_value=pad_value)

    # Logical view: unchanged shape, unchanged values.
    assert list(tt_output.shape) == list(shape)
    logical = ttnn.to_torch(tt_output).to(torch.float32)
    assert_with_pcc(torch_reference(torch_input), logical, PCC[ttnn.bfloat16])

    # Padded view: input in the leading region, pad_value everywhere else.
    padded = tt_output.cpu().to_torch_with_padded_shape().to(torch.float32)
    assert list(padded.shape) == list(target)

    expected = torch.full(target, float(pad_value), dtype=torch.float32)
    expected[..., : shape[-2], : shape[-1]] = torch_input
    assert_with_pcc(expected, padded, PCC[ttnn.bfloat16])


@pytest.mark.parametrize(
    "shape",
    [[1, 1, 30, 32], [1, 1, 32, 50], [1, 1, 50, 50]],
    ids=["h_non_aligned", "w_non_aligned", "hw_non_aligned"],
)
def test_tilize_auto_padding(device, shape):
    """`pad_mode="auto"`: the target is the input shape tile-rounded up."""
    torch_input, tt_input = _make_input(shape, ttnn.bfloat16, device)

    tt_output = tilize(tt_input, pad_value=0.0)

    assert list(tt_output.shape) == list(shape)
    logical = ttnn.to_torch(tt_output).to(torch.float32)
    assert_with_pcc(torch_reference(torch_input), logical, PCC[ttnn.bfloat16])

    target = list(shape[:-2]) + [((shape[-2] + 31) // 32) * 32, ((shape[-1] + 31) // 32) * 32]
    padded = tt_output.cpu().to_torch_with_padded_shape().to(torch.float32)
    assert list(padded.shape) == target

    expected = torch.zeros(target, dtype=torch.float32)
    expected[..., : shape[-2], : shape[-1]] = torch_input
    assert_with_pcc(expected, padded, PCC[ttnn.bfloat16])


def test_tilize_auto_padding_on_aligned_input_is_a_noop(device):
    """An already-aligned input under `pad_mode="auto"` must be a plain tilize."""
    torch_input, tt_input = _make_input([1, 1, 64, 64], ttnn.bfloat16, device)

    padded_out = tilize(tt_input, pad_value=0.0)
    plain_out = tilize(tt_input)

    assert list(padded_out.padded_shape) == list(plain_out.padded_shape)
    assert_with_pcc(
        ttnn.to_torch(plain_out).to(torch.float32),
        ttnn.to_torch(padded_out).to(torch.float32),
        PCC[ttnn.bfloat16],
    )


def test_tilize_program_cache(device):
    """Second call with identical arguments must hit the program cache."""
    torch_input, tt_input = _make_input([1, 1, 128, 128], ttnn.bfloat16, device)

    first = tilize(tt_input)
    entries_after_first = device.num_program_cache_entries()

    second = tilize(tt_input)
    entries_after_second = device.num_program_cache_entries()

    assert entries_after_second == entries_after_first, "second call compiled a new program"

    assert_with_pcc(
        torch_reference(torch_input),
        ttnn.to_torch(first).to(torch.float32),
        PCC[ttnn.bfloat16],
    )
    assert_with_pcc(
        torch_reference(torch_input),
        ttnn.to_torch(second).to(torch.float32),
        PCC[ttnn.bfloat16],
    )
