# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Element size (bytes) for each ttnn dtype we exercise in this file.
_DTYPE_ELEM_SIZE = {
    ttnn.bfloat16: 2,
    ttnn.float32: 4,
    ttnn.uint32: 4,
    ttnn.int32: 4,
    ttnn.uint16: 2,
    ttnn.uint8: 1,
    ttnn.bfloat8_b: 1,
}


def _divisible_grid_1d(total_dim, max_cores, step):
    """Return the largest n <= max_cores such that total_dim % (n * step) == 0.

    Raises:
        ValueError: If no valid n exists (i.e. total_dim is not a multiple of step).
    """
    for n in range(max_cores, 0, -1):
        if total_dim % (n * step) == 0:
            return n
    raise ValueError(f"No valid 1D grid size for total_dim={total_dim}, max_cores={max_cores}, step={step}")


def make_sharded_memory_config(device, shape, strategy, layout, dtype=ttnn.bfloat16):
    """Create a valid sharded MemoryConfig for `shape` on `device`.

    For TILE layout the shard dims must be tile-aligned (multiples of 32).
    For ROW_MAJOR, shard_width * element_size must be a multiple of the
    recommended L1 alignment (64 bytes today), so the shard-width step is
    derived from the `dtype` argument.
    """
    grid = device.compute_with_storage_grid_size()
    max_x, max_y = grid.x, grid.y
    tile_h, tile_w = 32, 32

    # TILE layout: pad last two dims to tile-multiples before collapsing so
    # total_h/total_w match the physical layout and create_sharded_memory_config
    # gets tile-aligned shards.
    shape_for_memcfg = list(shape)
    if layout == ttnn.TILE_LAYOUT and len(shape) >= 2:
        padded_h_dim = ((shape[-2] + tile_h - 1) // tile_h) * tile_h
        padded_w_dim = ((shape[-1] + tile_w - 1) // tile_w) * tile_w
        total_h = padded_h_dim
        for d in shape[:-2]:
            total_h *= d
        total_w = padded_w_dim
        shape_for_memcfg[-2] = padded_h_dim
        shape_for_memcfg[-1] = padded_w_dim
    else:
        total_h = 1
        for d in shape[:-1]:
            total_h *= d
        total_w = shape[-1]

    step_h = tile_h if layout == ttnn.TILE_LAYOUT else 1
    recommended_alignment_bytes = 64
    element_size = _DTYPE_ELEM_SIZE.get(dtype, 2)
    rm_step_w = max(1, recommended_alignment_bytes // element_size)
    step_w = tile_w if layout == ttnn.TILE_LAYOUT else rm_step_w

    if strategy == ttnn.ShardStrategy.HEIGHT:
        ny = _divisible_grid_1d(total_h, max_y, step_h)
        core_grid = ttnn.CoreGrid(y=ny, x=1)
    elif strategy == ttnn.ShardStrategy.WIDTH:
        nx = _divisible_grid_1d(total_w, max_x, step_w)
        core_grid = ttnn.CoreGrid(y=1, x=nx)
    else:  # BLOCK
        ny = _divisible_grid_1d(total_h, max_y, step_h)
        nx = _divisible_grid_1d(total_w, max_x, step_w)
        core_grid = ttnn.CoreGrid(y=ny, x=nx)

    return ttnn.create_sharded_memory_config(
        shape=shape_for_memcfg,
        core_grid=core_grid,
        strategy=strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


# ---------------------------------------------------------------------------
# Check pad_value behavior for tiled reshape.
# ---------------------------------------------------------------------------


def _read_padded_region(tensor):
    """Host flat view of the padded buffer via ``reshape(padded_shape, padded_shape)``.

    Inner-2D is tile-aligned, so no implicit padding fill run—read does not clobber test padding.
    """
    padded = ttnn.reshape(tensor, tensor.padded_shape, tensor.padded_shape)
    return ttnn.to_torch(padded)


@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        ((12,), (12, 1, 1)),
        ((32, 32), (32, 16, 2)),
        ((1, 96), (3, 1, 32)),
    ],
    ids=["issue_22178", "tile_to_sub_tile", "reshape_1d_to_3d"],
)
@pytest.mark.parametrize("pad_value", [0.0, 1.0, -2.5], ids=["pad_0", "pad_1", "pad_neg"])
def test_reshape_tiled_pad_value(device, input_shape, output_shape, pad_value):
    """Padded tile regions must be filled with pad_value (issue #22178)."""
    torch.manual_seed(0)
    torch_input = torch.arange(1, int(torch.tensor(input_shape).prod().item()) + 1, dtype=torch.float32).reshape(
        input_shape
    )

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.reshape(tt_input, output_shape, pad_value=pad_value)

    full = _read_padded_region(tt_output)

    # Logical volume preserved and values unchanged.
    logical = ttnn.to_torch(tt_output).reshape(-1)
    assert torch.equal(logical, torch_input.reshape(-1)), "Logical values changed during reshape"

    # Any element outside the logical volume must equal pad_value.
    total_padded = full.numel()
    total_logical = logical.numel()
    assert total_padded >= total_logical

    if total_padded > total_logical:
        padded_flat = full.reshape(-1)
        # Pad locations aren't guaranteed contiguous in the flat view; the aggregate
        # ``sum(padded) - sum(logical) == pad_count * pad_value`` catches uniform fill. For
        # float32, per-element spot checks would need a layout-accurate padding mask (omitted here).
        pad_count = total_padded - total_logical
        observed_pad_sum = padded_flat.sum().item() - logical.sum().item()
        expected_pad_sum = pad_count * pad_value
        assert abs(observed_pad_sum - expected_pad_sum) < 1e-3 * max(1.0, abs(expected_pad_sum)), (
            f"Padded region not filled with pad_value={pad_value}: "
            f"expected sum {expected_pad_sum}, observed {observed_pad_sum}"
        )


def test_reshape_tiled_default_pad_is_zero(device):
    """Without an explicit pad_value, padded tile regions must default to 0 (issue #22178)."""
    torch_input = torch.ones((12,), dtype=torch.float32)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = ttnn.reshape(tt_input, (12, 1, 1))
    full = _read_padded_region(tt_output)

    # Sum of the full padded buffer must equal sum of the logical input (since pad is 0).
    assert (
        abs(full.sum().item() - torch_input.sum().item()) < 1e-3
    ), "Default pad_value should be 0; padded region contains non-zero data"


@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        # tile-aligned input -> non-aligned output: exercises the fill on a clean input padding region.
        ([128, 128], [128, 4, 32]),
        # non-aligned input -> non-aligned output: verifies input padding lanes don't bleed into output fill.
        ([128, 96], [128, 3, 32]),
    ],
    ids=["aligned_in_padded_out", "padded_in_padded_out"],
)
@pytest.mark.parametrize(
    "strategy",
    [ttnn.ShardStrategy.HEIGHT],
    ids=["height"],
)
@pytest.mark.parametrize("pad_value", [0.0, 1.0, -2.5], ids=["pad_0", "pad_1", "pad_neg"])
def test_reshape_tiled_pad_value_sharded(device, input_shape, output_shape, strategy, pad_value):
    """Sharded tiled reshape with tile-padded output: exercises the s2i -> fill_pad -> i2s detour."""
    torch.manual_seed(0)
    torch_input = torch.arange(1, int(torch.tensor(input_shape).prod().item()) + 1, dtype=torch.bfloat16).reshape(
        input_shape
    )

    input_mem_cfg = make_sharded_memory_config(device, input_shape, strategy, ttnn.TILE_LAYOUT)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=input_mem_cfg,
    )

    tt_output = ttnn.reshape(tt_input, output_shape, pad_value=pad_value)

    # Logical values preserved.
    logical = ttnn.to_torch(tt_output).reshape(-1).to(torch.float32)
    assert torch.equal(logical, torch_input.reshape(-1).to(torch.float32)), "Logical values changed during reshape"

    # Padded-region check: sum(padded) - sum(logical) == pad_count * pad_value.
    full = _read_padded_region(tt_output).to(torch.float32)
    total_padded = full.numel()
    total_logical = logical.numel()
    assert total_padded > total_logical, "Test expects the output to have tile padding"
    pad_count = total_padded - total_logical
    observed_pad_sum = full.sum().item() - logical.sum().item()
    expected_pad_sum = pad_count * pad_value
    # bf16 accumulation is lossy for large buffers; scale tolerance with pad_count.
    tol = max(1e-2, 1e-2 * pad_count * max(1.0, abs(pad_value)))
    assert abs(observed_pad_sum - expected_pad_sum) < tol, (
        f"Padded region not filled with pad_value={pad_value} (strategy={strategy}): "
        f"expected sum {expected_pad_sum}, observed {observed_pad_sum}"
    )


# ---------------------------------------------------------------------------
# BFLOAT8_B pad_value path: fill runs on the bfloat16 intermediate before the
# typecast back to BFLOAT8_B; use PCC rather than exact equality.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        ((32, 32), (32, 16, 2)),
        ((1, 96), (3, 1, 32)),
    ],
    ids=["tile_to_sub_tile", "reshape_1d_to_3d"],
)
def test_reshape_tiled_pad_value_bfloat8_b(device, input_shape, output_shape):
    """BFLOAT8_B interleaved reshape: fill happens on the bf16 intermediate before re-typecast.

    Uses pad_value=0.0 because BF8 is a block-float format (16-element sub-blocks with shared exponent):
    a small pad_value in a sub-block that also contains large logical values quantizes to ~0, making
    exact-sum checks unreliable. 0.0 is represented exactly for any block exponent, so the check is clean.
    """
    torch.manual_seed(0)
    torch_input = torch.arange(1, int(torch.tensor(input_shape).prod().item()) + 1, dtype=torch.bfloat16).reshape(
        input_shape
    )

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.reshape(tt_input, output_shape, pad_value=0.0)

    # Logical values preserved within BF8 quantization (PCC).
    actual = ttnn.to_torch(tt_output).reshape(-1).to(torch.float32)
    expected_logical = torch_input.reshape(-1).to(torch.float32)
    assert_with_pcc(expected_logical, actual, 0.99)

    # With pad_value=0.0, padded lanes add no contribution: sum(full) ~= sum(logical).
    full = _read_padded_region(tt_output).to(torch.float32).reshape(-1)
    pad_count = full.numel() - actual.numel()
    if pad_count > 0:
        observed_pad_sum = full.sum().item() - actual.sum().item()
        # BF8 round-trip tolerance on the logical sum alone, scaled by pad count.
        tol = max(0.5, 0.01 * abs(actual.sum().item()))
        assert (
            abs(observed_pad_sum) < tol
        ), f"BF8 padded region not zero-filled: observed pad sum {observed_pad_sum} (tol {tol})"


# ---------------------------------------------------------------------------
# skip_padding_fill opt-out: callers that handle padding downstream can skip
# the fill_implicit_tile_padding dispatch.
# ---------------------------------------------------------------------------


def test_reshape_tiled_skip_padding_fill(device):
    """skip_padding_fill=True preserves logical values and bypasses the pad_value fill."""
    torch.manual_seed(0)
    torch_input = torch.arange(1, 13, dtype=torch.float32)  # shape (12,), tile-padded on device
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    # Opt-out path: logical region must be correct; padded lanes are caller's responsibility.
    tt_skip = ttnn.reshape(tt_input, (12, 1, 1), pad_value=42.0, skip_padding_fill=True)
    logical_skip = ttnn.to_torch(tt_skip).reshape(-1)
    assert torch.equal(logical_skip, torch_input), "skip_padding_fill=True must not corrupt logical values"

    # Sanity: default path (skip_padding_fill=False) still writes pad_value into padding.
    tt_filled = ttnn.reshape(tt_input, (12, 1, 1), pad_value=42.0)
    full = _read_padded_region(tt_filled).reshape(-1)
    pad_count = full.numel() - logical_skip.numel()
    assert pad_count > 0, "Test expects output to have tile padding"
    observed = full.sum().item() - logical_skip.sum().item()
    expected = pad_count * 42.0
    assert abs(observed - expected) < 1e-3 * max(
        1.0, abs(expected)
    ), f"Default path regressed: expected pad sum {expected}, observed {observed}"
