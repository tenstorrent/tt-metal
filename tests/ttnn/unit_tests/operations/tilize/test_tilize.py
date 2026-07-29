# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for `ttnn.operations.tilize.tilize`.

IMMUTABLE SPEC — the implementer must not modify this file.

tilize is a pure layout conversion: a ROW_MAJOR tensor is re-laid into TILE
layout (32x32 tiles of four 16x16 faces). Element VALUES are unchanged, so the
PyTorch reference is the identity function and the oracle is a round-trip
comparison:

    to_torch(tilize(from_torch(x, ROW_MAJOR))) ~= x

`dtype=` narrows the storage format, in which case the reference is `x` cast to
the output dtype and the comparison is PCC at the dtype's threshold.

Device comes from the root `device` fixture; `conftest.py` in this directory
promotes it to module scope. Do not open devices manually.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from ttnn.operations.tilize import tilize
from tests.ttnn.utils_for_testing import assert_with_pcc


# ---------------------------------------------------------------------------
# Reference + tolerances
# ---------------------------------------------------------------------------


def torch_tilize_reference(torch_tensor: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: tilize performs NO arithmetic. Layout changes; values do not."""
    return torch_tensor


# PCC keyed by the dtype the result is stored in.
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,  # bf8b reads back as bf16
    ttnn.uint32: torch.int32,
    ttnn.uint16: torch.int32,
    ttnn.int32: torch.int32,
}


def _make_input(shape, dtype):
    torch.manual_seed(42)
    if dtype in (ttnn.uint32, ttnn.uint16):
        return torch.randint(0, 1000, shape, dtype=torch.int32)
    if dtype == ttnn.int32:
        return torch.randint(-1000, 1000, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.randn(shape, dtype=torch.float32)
    return torch.randn(shape).bfloat16()


def _run(
    device,
    shape,
    *,
    dtype=ttnn.bfloat16,
    output_dtype=None,
    use_multicore=True,
    use_double_buffer=True,
    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    output_memory_config=None,
):
    """Build a ROW_MAJOR input, tilize it, read back, and return (actual, expected)."""
    torch_input = _make_input(shape, dtype)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_memory_config,
    )

    tt_output = tilize(
        tt_input,
        output_memory_config,
        dtype=output_dtype,
        use_multicore=use_multicore,
        use_double_buffer=use_double_buffer,
    )

    assert tt_output.layout == ttnn.TILE_LAYOUT, f"expected TILE_LAYOUT, got {tt_output.layout}"
    assert list(tt_output.shape) == list(shape), f"shape changed: {tt_output.shape} vs {shape}"

    out_dtype = output_dtype if output_dtype is not None else dtype
    assert tt_output.dtype == out_dtype, f"expected dtype {out_dtype}, got {tt_output.dtype}"

    actual = ttnn.to_torch(tt_output)
    expected = torch_tilize_reference(torch_input).to(_TORCH_DTYPE[out_dtype])
    return actual, expected, out_dtype


# ---------------------------------------------------------------------------
# 1. Core identity — shapes x dtypes x single/multi core
# ---------------------------------------------------------------------------

SHAPES = [
    pytest.param((1, 1, 32, 32), id="single_tile"),
    pytest.param((1, 1, 64, 128), id="multi_tile"),
    pytest.param((1, 1, 96, 64), id="non_square"),
    pytest.param((2, 3, 64, 96), id="multi_batch"),
    pytest.param((1, 1, 32, 4096), id="wide_short"),
    pytest.param((1, 1, 2048, 64), id="tall_narrow"),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("use_multicore", [False, True], ids=["single_core", "multi_core"])
def test_tilize(device, shape, dtype, use_multicore):
    actual, expected, out_dtype = _run(device, shape, dtype=dtype, use_multicore=use_multicore)
    assert_with_pcc(expected, actual, PCC[out_dtype])


# ---------------------------------------------------------------------------
# 2. Rank 2 / 3 / 4
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((64, 128), id="rank2"),
        pytest.param((2, 32, 64), id="rank3"),
        pytest.param((1, 2, 64, 64), id="rank4"),
    ],
)
@pytest.mark.parametrize("use_multicore", [False, True], ids=["single_core", "multi_core"])
def test_tilize_rank(device, shape, use_multicore):
    actual, expected, out_dtype = _run(device, shape, use_multicore=use_multicore)
    assert_with_pcc(expected, actual, PCC[out_dtype])


# ---------------------------------------------------------------------------
# 3. Value-preserving output-dtype cast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,output_dtype",
    [
        pytest.param(ttnn.bfloat16, ttnn.bfloat16, id="bf16_to_bf16"),
        pytest.param(ttnn.bfloat16, ttnn.float32, id="bf16_to_fp32"),
        pytest.param(ttnn.float32, ttnn.bfloat16, id="fp32_to_bf16"),
        pytest.param(ttnn.float32, ttnn.float32, id="fp32_to_fp32"),
        pytest.param(ttnn.bfloat16, ttnn.bfloat8_b, id="bf16_to_bf8b"),
        pytest.param(ttnn.float32, ttnn.bfloat8_b, id="fp32_to_bf8b"),
    ],
)
def test_tilize_output_dtype(device, dtype, output_dtype):
    actual, expected, out_dtype = _run(device, (1, 1, 64, 128), dtype=dtype, output_dtype=output_dtype)
    assert_with_pcc(expected.float(), actual.float(), PCC[out_dtype])


# ---------------------------------------------------------------------------
# 4. Integer passthrough — exact, no cast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [ttnn.uint32, ttnn.uint16, ttnn.int32], ids=["uint32", "uint16", "int32"])
def test_tilize_integer_passthrough(device, dtype):
    actual, expected, _ = _run(device, (1, 1, 64, 128), dtype=dtype)
    assert torch.equal(actual.to(torch.int32), expected.to(torch.int32)), "integer tilize must be bit-exact"


# ---------------------------------------------------------------------------
# 5. Memory configs — interleaved DRAM / L1, both directions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_memory_config,output_memory_config",
    [
        pytest.param(ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, id="dram_to_dram"),
        pytest.param(ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, id="dram_to_l1"),
        pytest.param(ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, id="l1_to_l1"),
        pytest.param(ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, id="l1_to_dram"),
    ],
)
def test_tilize_memory_config(device, input_memory_config, output_memory_config):
    actual, expected, out_dtype = _run(
        device,
        (1, 1, 64, 128),
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
    )
    assert_with_pcc(expected, actual, PCC[out_dtype])


def test_tilize_default_memory_config_follows_input(device):
    """`memory_config=None` must inherit the input's memory config."""
    torch.manual_seed(42)
    torch_input = torch.randn((1, 1, 64, 128)).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_output = tilize(tt_input)
    assert tt_output.memory_config().buffer_type == ttnn.BufferType.L1
    assert_with_pcc(torch_input, ttnn.to_torch(tt_output), PCC[ttnn.bfloat16])


# ---------------------------------------------------------------------------
# 6. Single-buffered opt-out — identity must be unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_double_buffer", [False, True], ids=["depth1", "depth2"])
@pytest.mark.parametrize("use_multicore", [False, True], ids=["single_core", "multi_core"])
def test_tilize_double_buffer(device, use_double_buffer, use_multicore):
    actual, expected, out_dtype = _run(
        device,
        (1, 1, 64, 2048),
        use_double_buffer=use_double_buffer,
        use_multicore=use_multicore,
    )
    assert_with_pcc(expected, actual, PCC[out_dtype])


# ---------------------------------------------------------------------------
# 7. Sharded I/O — L1 shard in, L1 shard out (same spec, zero-copy path)
# ---------------------------------------------------------------------------


def _shard_config(scheme, grid, shard_shape, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    return ttnn.MemoryConfig(
        scheme,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard_shape, orientation),
    )


def _line_grid(n):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n - 1, 0))})


def _box_grid(n):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n - 1, n - 1))})


@pytest.mark.parametrize(
    "shape,scheme,grid,shard_shape",
    [
        pytest.param(
            (1, 1, 512, 64),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            _line_grid(4),
            (128, 64),
            id="height_sharded",
        ),
        pytest.param(
            (1, 1, 64, 512),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            _line_grid(4),
            (64, 128),
            id="width_sharded",
        ),
        pytest.param(
            (1, 1, 128, 128),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            _box_grid(2),
            (64, 64),
            id="block_sharded",
        ),
    ],
)
def test_tilize_sharded_same_spec(device, shape, scheme, grid, shard_shape):
    mem_config = _shard_config(scheme, grid, shard_shape)
    actual, expected, out_dtype = _run(
        device,
        shape,
        input_memory_config=mem_config,
        output_memory_config=mem_config,
    )
    assert_with_pcc(expected, actual, PCC[out_dtype])


def test_tilize_interleaved_to_sharded(device):
    """DRAM-interleaved ROW_MAJOR in -> L1 HEIGHT-sharded TILE out (split-reader path)."""
    mem_config = _shard_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _line_grid(4), (32, 64))
    actual, expected, out_dtype = _run(
        device,
        (1, 1, 128, 64),
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_memory_config=mem_config,
    )
    assert_with_pcc(expected, actual, PCC[out_dtype])


def test_tilize_sharded_to_interleaved(device):
    """L1 HEIGHT-sharded ROW_MAJOR in -> DRAM-interleaved TILE out (split-writer path)."""
    mem_config = _shard_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _line_grid(4), (32, 64))
    actual, expected, out_dtype = _run(
        device,
        (1, 1, 128, 64),
        input_memory_config=mem_config,
        output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    assert_with_pcc(expected, actual, PCC[out_dtype])


def test_tilize_nd_sharded(device):
    """NdShardSpec in -> NdShardSpec out, same spec."""
    grid = _box_grid(2)
    nd_spec = ttnn.NdShardSpec(ttnn.Shape((1, 1, 64, 64)), grid, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.BufferType.L1, nd_spec)
    actual, expected, out_dtype = _run(
        device,
        (1, 1, 128, 128),
        input_memory_config=mem_config,
        output_memory_config=mem_config,
    )
    assert_with_pcc(expected, actual, PCC[out_dtype])


# ---------------------------------------------------------------------------
# 8. Program cache — second call with the same spec must hit
# ---------------------------------------------------------------------------


def test_tilize_program_cache(device):
    shape = (1, 1, 64, 128)
    torch.manual_seed(42)
    torch_input = torch.randn(shape).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    first = tilize(tt_input)
    entries_after_first = device.num_program_cache_entries()

    second = tilize(tt_input)
    entries_after_second = device.num_program_cache_entries()

    assert entries_after_second == entries_after_first, (
        "second tilize with an identical shape/dtype/memory_config must hit the program cache "
        f"(entries went {entries_after_first} -> {entries_after_second})"
    )
    assert_with_pcc(torch_input, ttnn.to_torch(first), PCC[ttnn.bfloat16])
    assert_with_pcc(torch_input, ttnn.to_torch(second), PCC[ttnn.bfloat16])


# ---------------------------------------------------------------------------
# 9. Validation
# ---------------------------------------------------------------------------


def test_tilize_rejects_tile_layout_input(device, expect_error):
    torch.manual_seed(42)
    tt_input = ttnn.from_torch(
        torch.randn((1, 1, 32, 32)).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    with expect_error(RuntimeError, "ROW_MAJOR"):
        tilize(tt_input)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 47), id="W_not_div_32"),
        pytest.param((1, 1, 47, 32), id="H_not_div_32"),
        pytest.param((1, 1, 30, 30), id="both_not_div_32"),
    ],
)
def test_tilize_rejects_unaligned_shape(device, expect_error, shape):
    torch.manual_seed(42)
    tt_input = ttnn.from_torch(
        torch.randn(shape).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    with expect_error(ValueError, "divisible by 32"):
        tilize(tt_input)


def test_tilize_rejects_host_tensor(expect_error):
    torch.manual_seed(42)
    host_tensor = ttnn.from_torch(
        torch.randn((1, 1, 32, 32)).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    with expect_error(RuntimeError, "device"):
        tilize(host_tensor)
