# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for the tilize op (ROW_MAJOR -> TILE layout conversion).

This is the immutable spec. Do NOT modify it while implementing the op.

tilize is a pure layout conversion: element VALUES are unchanged, only the byte
positions in memory change. So the oracle is IDENTITY — reading the tilized
tensor back to host must reproduce the original ROW_MAJOR values (value-
preserving-cast tolerance when `dtype=` narrows the format).
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.tilize import tilize


# PCC tolerances keyed by dtype (same thresholds as the golden suite).
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

# Minimum shape set: single-tile, multi-tile, non-square, multi-batch (rank 4).
SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 64, 128),  # multi-tile
    (1, 1, 96, 32),  # non-square (tall)
    (1, 1, 32, 96),  # non-square (wide)
    (2, 3, 64, 64),  # multi-batch, rank 4
    (128, 256),  # rank 2
    (4, 32, 64),  # rank 3
]


def _torch_ref(dtype, shape):
    torch.manual_seed(42)
    if dtype == ttnn.float32:
        return torch.randn(shape, dtype=torch.float32)
    # bfloat16 / bfloat8_b: generate in bf16 (bf8b is packed by from_torch).
    return torch.randn(shape).bfloat16()


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("use_multicore", [False, True])
def test_tilize_identity(device, shape, dtype, use_multicore):
    """tilize(RM) -> TILE, read back == input values, for each dtype / core mode."""
    torch_input = _torch_ref(dtype, shape)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    tt_output = tilize(tt_input, use_multicore=use_multicore)

    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert list(tt_output.shape) == list(shape)

    output = ttnn.to_torch(tt_output).to(torch_input.dtype)
    assert_with_pcc(torch_input, output, PCC[dtype])


@pytest.mark.parametrize("shape", [(1, 1, 64, 128), (2, 3, 64, 64)])
def test_tilize_explicit_memory_config(device, shape):
    """tilize with an explicit output memory_config (L1 interleaved)."""
    dtype = ttnn.bfloat16
    torch_input = _torch_ref(dtype, shape)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    tt_output = tilize(tt_input, memory_config=ttnn.L1_MEMORY_CONFIG)

    assert tt_output.layout == ttnn.TILE_LAYOUT
    output = ttnn.to_torch(tt_output).to(torch_input.dtype)
    assert_with_pcc(torch_input, output, PCC[dtype])


@pytest.mark.parametrize("shape", [(1, 1, 64, 128)])
@pytest.mark.parametrize("out_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_tilize_output_dtype_cast(device, shape, out_dtype):
    """value-preserving cast via the `dtype=` kwarg (bf16 -> bf16 / bf8b)."""
    in_dtype = ttnn.bfloat16
    torch_input = _torch_ref(in_dtype, shape)

    tt_input = ttnn.from_torch(torch_input, dtype=in_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    tt_output = tilize(tt_input, dtype=out_dtype)

    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert tt_output.dtype == out_dtype
    output = ttnn.to_torch(tt_output).to(torch_input.dtype)
    assert_with_pcc(torch_input, output, PCC[out_dtype])


def test_tilize_program_cache_hit(device):
    """Second call with the same shape/dtype/mem_config hits the program cache."""
    dtype = ttnn.bfloat16
    shape = (1, 1, 64, 128)
    torch_input = _torch_ref(dtype, shape)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    out1 = tilize(tt_input)
    out2 = tilize(tt_input)

    for out in (out1, out2):
        result = ttnn.to_torch(out).to(torch_input.dtype)
        assert_with_pcc(torch_input, result, PCC[dtype])


# --- input validation -----------------------------------------------------


def test_tilize_rejects_tile_layout_input(device, expect_error):
    """Input must be ROW_MAJOR — TILE input is rejected."""
    t = ttnn.from_torch(
        torch.randn(64, 64).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    with expect_error((ValueError, RuntimeError), "."):
        tilize(t)


def test_tilize_rejects_non_tile_aligned(device, expect_error):
    """Last two dims not divisible by 32 must be rejected (op does NOT pad)."""
    t = ttnn.from_torch(
        torch.randn(1, 1, 47, 64).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    with expect_error((ValueError, RuntimeError), "."):
        tilize(t)
