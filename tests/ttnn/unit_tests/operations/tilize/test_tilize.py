# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for tilize — the immutable Phase 0 spec.

Phase 0 corner (op_design.md -> "Phase 0 SUPPORTED rectangle"): bfloat16 in and
out, TensorMemoryLayout::INTERLEAVED DRAM in and out, rank 4, tile-aligned, no
padding, 32x32 output tiles, low_l1=False, tile_grid in {single_tile, small,
tall_narrow}.

tilize is a pure re-lay of bytes, so the reference is the identity:
    to_torch(tilize(from_torch(x, ROW_MAJOR))) == x
bfloat16 -> bfloat16 is exactly representable, so the comparison is exact
(the golden suite's TOLERANCES hold bf16 to "exact" as well); PCC is reported
alongside for diagnostics only.

Device is module-scoped by this directory's conftest.py (use_module_device).
"""

import pytest
import torch
import ttnn

from ttnn.operations.tilize import tilize


# PCC floors keyed by output dtype — the same thresholds as the golden suite.
# Only used on a lossy cast; Phase 0 (bf16 -> bf16) is exact.
PCC_BY_DTYPE = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}
EXACT_DTYPES = (ttnn.bfloat16, ttnn.float32)


def pytorch_reference(x: torch.Tensor) -> torch.Tensor:
    """tilize changes addresses, not values or logical positions."""
    return x.clone()


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    if torch.equal(a, b):
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom != 0 else 0.0


def _make_rm_input(shape, device, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    return torch_input, tt_input


def _check(tt_output, torch_input, *, shape, dtype=ttnn.bfloat16):
    # Layout first: correct values still in ROW_MAJOR would be a broken tilize.
    assert tt_output.layout == ttnn.TILE_LAYOUT, f"layout {tt_output.layout} != TILE_LAYOUT"
    assert tt_output.dtype == dtype, f"dtype {tt_output.dtype} != {dtype}"
    tile_shape = list(tt_output.tile.tile_shape)
    assert tile_shape == [32, 32], f"tile shape {tile_shape} != [32, 32]"
    # Logical shape is unchanged.
    assert list(tt_output.shape) == list(shape), f"shape {list(tt_output.shape)} != {list(shape)}"
    mc = tt_output.memory_config()
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert mc.buffer_type == ttnn.BufferType.DRAM

    torch_output = ttnn.to_torch(tt_output)
    expected = pytorch_reference(torch_input)
    assert list(torch_output.shape) == list(expected.shape)
    if dtype in EXACT_DTYPES:
        mismatches = int((torch_output.to(expected.dtype) != expected).sum())
        assert mismatches == 0, (
            f"tilize must be bit-identical: {mismatches} of {expected.numel()} elements differ "
            f"(pcc={_pcc(torch_output.float(), expected.float()):.6f})"
        )
    else:
        pcc = _pcc(torch_output.float(), expected.float())
        assert pcc >= PCC_BY_DTYPE[dtype], f"pcc {pcc} < {PCC_BY_DTYPE[dtype]}"


# Every shape is rank 4, tile-aligned, and in tile_grid {single_tile, small,
# tall_narrow}. R = prod(shape[:-2]) * H/32, C = W/32.
PHASE0_SHAPES = [
    pytest.param((1, 1, 32, 32), id="single_tile"),  # R=1,  C=1  single_tile
    pytest.param((1, 1, 64, 128), id="multi_tile_small"),  # R=2,  C=4  small
    pytest.param((1, 1, 96, 64), id="non_square_small"),  # R=3,  C=2  small
    pytest.param((2, 3, 64, 96), id="multi_batch_small"),  # R=12, C=3  small (leading-dim fold)
    pytest.param((1, 1, 1024, 32), id="tall_narrow_one_col"),  # R=32, C=1  tall_narrow
    pytest.param((4, 1, 256, 32), id="tall_narrow_multi_batch"),  # R=32, C=1  tall_narrow via the fold
    pytest.param((1, 1, 2048, 64), id="tall_narrow_grid_scale"),  # R=64, C=2  tall_narrow
    pytest.param((1, 1, 16384, 64), id="tall_narrow_perf_focus"),  # R=512, C=2 tall_narrow (many rows per core)
    pytest.param((1, 1, 1600, 96), id="tall_narrow_uneven_split"),  # R=50, C=3 tall_narrow (ragged row split)
]


@pytest.mark.parametrize("shape", PHASE0_SHAPES)
def test_tilize(device, shape):
    torch_input, tt_input = _make_rm_input(shape, device)
    tt_output = tilize(tt_input)
    _check(tt_output, torch_input, shape=shape)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 64), id="baseline"),
        pytest.param((1, 1, 2048, 64), id="tall_narrow"),
    ],
)
def test_tilize_explicit_kwargs(device, shape):
    """Explicit memory_config= and a no-cast dtype= are the Phase 0 call shapes."""
    torch_input, tt_input = _make_rm_input(shape, device)
    tt_output = tilize(tt_input, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    _check(tt_output, torch_input, shape=shape)


def test_tilize_input_untouched(device):
    """A re-lay into a new output must not modify its input."""
    shape = (1, 1, 64, 128)
    torch_input, tt_input = _make_rm_input(shape, device)
    _ = tilize(tt_input)
    assert tt_input.layout == ttnn.ROW_MAJOR_LAYOUT
    assert torch.equal(ttnn.to_torch(tt_input), torch_input)


def test_tilize_program_cache(device):
    """Same config on fresh allocations: at most one program, then cache hits."""
    shape = (1, 1, 2048, 64)
    device.enable_program_cache()
    n0 = device.num_program_cache_entries()
    keep_alive = []
    first_delta = None
    for i in range(3):
        torch_input, tt_input = _make_rm_input(shape, device)
        tt_output = tilize(tt_input)
        keep_alive.extend([tt_input, tt_output])
        _check(tt_output, torch_input, shape=shape)
        delta = device.num_program_cache_entries() - n0
        if i == 0:
            first_delta = delta
            assert first_delta <= 1, f"first call built {first_delta} programs"
        else:
            assert delta == first_delta, f"call {i + 1} added {delta - first_delta} program(s)"


# --- validation: malformed calls raise ValueError / RuntimeError -------------
# The message is not pinned ("" matches any text). NotImplementedError, a
# RuntimeError subclass, would also be accepted, but these are malformed inputs
# and the op's entry point refuses them before the support gate (op_design.md
# -> validation order).


def test_tilize_rejects_host_tensor(expect_error):
    torch.manual_seed(42)
    x = torch.randn((1, 1, 32, 32), dtype=torch.float32).to(torch.bfloat16)
    host = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    with expect_error((ValueError, RuntimeError), ""):
        tilize(host)


def test_tilize_rejects_tile_input_without_tile(device, expect_error):
    torch.manual_seed(42)
    x = torch.randn((1, 1, 32, 64), dtype=torch.float32).to(torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    with expect_error((ValueError, RuntimeError), ""):
        tilize(t)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 50, 64), id="h_not_multiple_of_32"),
        pytest.param((1, 1, 32, 50), id="w_not_multiple_of_32"),
        pytest.param((1, 1, 50, 50), id="both_unaligned"),
    ],
)
def test_tilize_rejects_unaligned_without_padding(device, shape, expect_error):
    _, tt_input = _make_rm_input(shape, device)
    with expect_error((ValueError, RuntimeError), ""):
        tilize(tt_input)


def test_tilize_rejects_small_output_padded_shape(device, expect_error):
    _, tt_input = _make_rm_input((1, 1, 64, 64), device)
    with expect_error((ValueError, RuntimeError), ""):
        tilize(tt_input, output_padded_shape=[1, 1, 32, 64], pad_value=0)


@pytest.mark.parametrize(
    "tile_hw",
    [
        pytest.param([24, 32], id="height_not_pow2_fraction"),
        pytest.param([64, 32], id="height_above_32"),
        pytest.param([32, 16], id="width_not_32"),
    ],
)
def test_tilize_rejects_bad_tile(device, tile_hw, expect_error):
    _, tt_input = _make_rm_input((1, 1, 64, 64), device)
    with expect_error((ValueError, RuntimeError), ""):
        # ttnn.Tile itself may refuse some shapes; either refusal satisfies the spec.
        tilize(tt_input, tile=ttnn.Tile(tile_hw))
