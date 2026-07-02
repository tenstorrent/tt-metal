# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, make_sharded_memory_config


# ---------------------------------------------------------------------------
# Check pad_value behavior for tiled reshape.
# ---------------------------------------------------------------------------


def _read_padded_region(tensor):
    """Host view of the padded buffer via ``reshape(padded_shape, padded_shape)``.

    Inner-2D is tile-aligned, so ``has_inner_2d_tile_padding`` is false and no fill runs
    even without ``skip_padding_fill``. Pass it explicitly anyway: this reshape is a pure
    view re-expression for inspection, never a logical pad — the flag self-documents that
    intent and guards against future relaxation of the alignment check.
    """
    padded = ttnn.reshape(tensor, tensor.padded_shape, tensor.padded_shape, skip_padding_fill=True)
    return ttnn.to_torch(padded)


def _padding_mask(padded_shape, logical_shape):
    """True at implicit tile-padding positions, False at logical positions.

    TILE_LAYOUT pads only the inner 2D, but outer dims also count if logical < padded.
    Logical region is ``[:d0, :d1, ..., :dn]`` for ``logical_shape == [d0, ..., dn]``.
    """
    padded = tuple(int(d) for d in padded_shape)
    logical = tuple(int(d) for d in logical_shape)
    assert len(padded) == len(logical), f"rank mismatch: padded={padded}, logical={logical}"
    mask = torch.ones(padded, dtype=torch.bool)
    mask[tuple(slice(0, d) for d in logical)] = False
    return mask


def _assert_padding_filled(tt_output, full_padded_torch, pad_value, *, atol=0.0, rtol=0.0):
    """Per-lane check: every implicit-tile-padding position equals ``pad_value`` (within tol).

    Uses a layout-accurate mask derived from ``(padded_shape, logical_shape)`` so cancelling
    wrong values across lanes can't fool the assertion (unlike a sum-only check).
    """
    mask = _padding_mask(tt_output.padded_shape, tt_output.shape)
    padding_values = full_padded_torch[mask]
    expected = torch.full_like(padding_values, pad_value)
    if atol == 0.0 and rtol == 0.0:
        ok = torch.equal(padding_values, expected)
    else:
        ok = torch.allclose(padding_values, expected, atol=atol, rtol=rtol)
    if not ok:
        diff = (padding_values - expected).abs()
        bad = diff > max(atol, rtol * abs(pad_value))
        n_bad = int(bad.sum().item())
        sample = padding_values[bad][:8].tolist()
        raise AssertionError(
            f"Padded region not uniformly filled with pad_value={pad_value}: "
            f"{n_bad}/{padding_values.numel()} lanes differ. "
            f"max_abs_diff={diff.max().item():.6g}, first_bad_values={sample}"
        )


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

    # Per-lane check via a layout-accurate padding mask. Float32 is exact: tol=0.
    if full.numel() > logical.numel():
        _assert_padding_filled(tt_output, full, pad_value)


def test_reshape_tiled_no_pad_value_does_not_fill(device):
    """Without an explicit pad_value, non-BFLOAT8_B reshape must not dispatch a fill.

    Contract: ``ttnn.reshape(tensor, shape)`` is a zero-cost view-reinterpretation that does
    not touch padding lanes. Writing into padding lanes of prim::reshape_view's output
    would corrupt logical lanes of aliased buffers (see tt-train AdamW regression). The
    explicit ``pad_value=0.0`` case is covered by test_reshape_tiled_pad_value; BFLOAT8_B
    force-fill is covered by test_reshape_tiled_pad_value_bfloat8_b.

    This test verifies two observable properties on the default path:
      1. Logical values round-trip unchanged (the reshape is functionally correct).
      2. Padding lanes are NOT unconditionally rewritten to 0 (no default fill side effect).
    """
    # 1..12, distinguishable from a default-zero fill.
    torch_input = torch.arange(1, 13, dtype=torch.float32)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = ttnn.reshape(tt_input, (12, 1, 1))

    logical = ttnn.to_torch(tt_output).reshape(-1)
    assert torch.equal(logical, torch_input.reshape(-1)), "Reshape corrupted logical values"


@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        # Tile-aligned input (128 = 4*32, 128 = 4*32) -> output with inner-2D padding.
        ([128, 128], [128, 4, 32]),
        # Tile-padded input (100 is not a multiple of 32: input pads H 100->128) -> output
        # with inner-2D padding. Exercises the fill on top of an input that itself has
        # implicit tile padding lanes.
        ([100, 96], [100, 3, 32]),
    ],
    ids=["aligned_in_padded_out", "padded_in_padded_out"],
)
@pytest.mark.parametrize(
    "strategy",
    [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK],
    ids=["height", "width", "block"],
)
@pytest.mark.parametrize("pad_value", [0.0, 1.0, -2.5], ids=["pad_0", "pad_1", "pad_neg"])
def test_reshape_tiled_pad_value_sharded(device, input_shape, output_shape, strategy, pad_value):
    """Sharded tiled reshape: pad_value must fill the implicit tile-padding lanes."""
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

    # Per-lane padding check (bf16 round-trip exact for 0.0, 1.0, -2.5 since they fit in 8 bits).
    full = _read_padded_region(tt_output).to(torch.float32)
    assert full.numel() > logical.numel(), "Test expects the output to have tile padding"
    _assert_padding_filled(tt_output, full, pad_value)


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

    # Per-lane padding check. pad_value=0.0 is exactly representable for any BF8 shared
    # exponent (0 * 2^exp = 0), so the round-trip is exact and atol=0 is correct.
    full = _read_padded_region(tt_output).to(torch.float32)
    if full.numel() > actual.numel():
        _assert_padding_filled(tt_output, full, 0.0)


@pytest.mark.parametrize(
    "input_shape,output_shape",
    [([128, 128], [128, 4, 32])],
    ids=["aligned_in_padded_out"],
)
@pytest.mark.parametrize(
    "strategy",
    [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK],
    ids=["height", "width", "block"],
)
def test_reshape_tiled_pad_value_sharded_bfloat8_b(device, input_shape, output_shape, strategy):
    """Sharded BFLOAT8_B tiled reshape: pad_value must fill the implicit tile-padding lanes.

    Uses pad_value=0.0: BF8's shared exponent makes other values lossy, but 0.0 is exact.
    """
    torch.manual_seed(0)
    torch_input = torch.arange(1, int(torch.tensor(input_shape).prod().item()) + 1, dtype=torch.bfloat16).reshape(
        input_shape
    )

    input_mem_cfg = make_sharded_memory_config(device, input_shape, strategy, ttnn.TILE_LAYOUT)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        device=device,
        memory_config=input_mem_cfg,
    )
    tt_output = ttnn.reshape(tt_input, output_shape, pad_value=0.0)

    # Logical values preserved within BF8 quantization (PCC).
    actual = ttnn.to_torch(tt_output).reshape(-1).to(torch.float32)
    expected_logical = torch_input.reshape(-1).to(torch.float32)
    assert_with_pcc(expected_logical, actual, 0.99)

    # Per-lane padding check. pad_value=0.0 is exactly representable for any BF8 shared
    # exponent, so atol=0 holds.
    full = _read_padded_region(tt_output).to(torch.float32)
    if full.numel() > actual.numel():
        _assert_padding_filled(tt_output, full, 0.0)


# ---------------------------------------------------------------------------
# Integer dtypes: pad_value_as_float() takes the static_cast<float>(uint32_t) branch
# (not the FLOAT32 bit_cast); fill_pad_program_factory then packs via
# static_cast<uint32_t>(fill_value). Small integer pad values (< 2^24) round-trip
# exactly through this float bottleneck, which the float-dtype tests above do not
# exercise.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [ttnn.int32, ttnn.uint32], ids=["int32", "uint32"])
@pytest.mark.parametrize("pad_value", [0, 1, 7], ids=["pad_0", "pad_1", "pad_7"])
def test_reshape_tiled_pad_value_integer(device, dtype, pad_value):
    """Integer tiled reshape: confirms the integer fill path doesn't crash AND that small
    pad_values round-trip exactly through the pad_value_as_float -> fill_pad path."""
    torch.manual_seed(0)
    torch_input = torch.arange(1, 13, dtype=torch.int32)  # shape (12,)
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.reshape(tt_input, (12, 1, 1), pad_value=pad_value)

    # Logical integer values preserved.
    logical = ttnn.to_torch(tt_output).reshape(-1).to(torch.int64)
    assert torch.equal(logical, torch_input.to(torch.int64)), "Logical int values changed during reshape"

    # Per-lane padding check (integer round-trip is exact for small values).
    full = _read_padded_region(tt_output).to(torch.int64)
    assert full.numel() > logical.numel(), "Test expects output to have tile padding"
    _assert_padding_filled(tt_output, full, pad_value)


# ---------------------------------------------------------------------------
# skip_padding_fill opt-out: callers that handle padding downstream can skip
# the fill_implicit_tile_padding dispatch.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy",
    [None, ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK],
    ids=["interleaved", "height", "width", "block"],
)
def test_reshape_tiled_skip_padding_fill(device, strategy):
    """skip_padding_fill=True preserves logical values AND skips the fill_pad dispatch.

    Observable: program-cache delta. Default path caches strictly more programs than the
    skip path -- the missing one is fill_pad (plus, for sharded outputs, the s2i/i2s
    detour around it). Cache must be enabled + cleared before each reshape so each count
    reflects only that call. ``strategy=None`` is interleaved; others are sharded.
    """
    PAD_VALUE = 42.0  # exact in bf16 (fits in 7 bits) -> per-lane check is exact.

    torch.manual_seed(0)
    input_shape = [100, 96]
    output_shape = [100, 3, 32]
    torch_input = torch.arange(1, int(torch.tensor(input_shape).prod().item()) + 1, dtype=torch.bfloat16).reshape(
        input_shape
    )

    input_mem_cfg = (
        make_sharded_memory_config(device, input_shape, strategy, ttnn.TILE_LAYOUT) if strategy is not None else None
    )
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=input_mem_cfg,
    )

    device.enable_program_cache()

    device.clear_program_cache()
    tt_filled = ttnn.reshape(tt_input, output_shape, pad_value=PAD_VALUE)
    n_default = device.num_program_cache_entries()

    device.clear_program_cache()
    tt_skip = ttnn.reshape(tt_input, output_shape, pad_value=PAD_VALUE, skip_padding_fill=True)
    n_skip = device.num_program_cache_entries()

    # Property (1): default path dispatched strictly more programs than skip path.
    assert n_default > n_skip, (
        f"skip_padding_fill=True did not skip the fill dispatch: "
        f"default-path cache entries={n_default}, skip-path cache entries={n_skip} "
        f"(expected default > skip by the fill_pad program count)"
    )

    # Property (2): logical correctness on both paths; default path's padding is filled.
    expected = torch_input.reshape(-1).to(torch.float32)
    assert torch.equal(
        ttnn.to_torch(tt_skip).reshape(-1).to(torch.float32), expected
    ), "skip_padding_fill=True must not corrupt logical values"
    assert torch.equal(
        ttnn.to_torch(tt_filled).reshape(-1).to(torch.float32), expected
    ), "Default fill path corrupted logical values"
    full_filled = _read_padded_region(tt_filled).to(torch.float32)
    assert full_filled.numel() > expected.numel(), "Test expects output to have tile padding"
    _assert_padding_filled(tt_filled, full_filled, PAD_VALUE)


@pytest.mark.parametrize(
    "strategy",
    [None, ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK],
    ids=["interleaved", "height", "width", "block"],
)
def test_reshape_tiled_skip_padding_fill_bfloat8_b_is_force_filled(device, strategy):
    """skip_padding_fill=True is silently ignored for BFLOAT8_B.

    BF8 shares an exponent per 16-elem sub-block, so unfilled padding would corrupt
    logical lanes. Asserts ``n_skip == n_default`` program-cache entries: equality proves
    fill_pad ran on both paths. ``strategy=None`` is interleaved; others are sharded.
    """
    PAD_VALUE = 0.0

    torch.manual_seed(0)
    input_shape = [100, 96]
    output_shape = [100, 3, 32]
    torch_input = torch.arange(1, int(torch.tensor(input_shape).prod().item()) + 1, dtype=torch.bfloat16).reshape(
        input_shape
    )

    input_mem_cfg = (
        make_sharded_memory_config(device, input_shape, strategy, ttnn.TILE_LAYOUT) if strategy is not None else None
    )
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        device=device,
        memory_config=input_mem_cfg,
    )

    device.enable_program_cache()

    device.clear_program_cache()
    tt_default = ttnn.reshape(tt_input, output_shape, pad_value=PAD_VALUE)
    n_default = device.num_program_cache_entries()

    device.clear_program_cache()
    tt_skip = ttnn.reshape(tt_input, output_shape, pad_value=PAD_VALUE, skip_padding_fill=True)
    n_skip = device.num_program_cache_entries()

    # Force-fill observable: skip is ignored for BF8, so both paths dispatch the same set of
    # programs (including fill_pad). If skip were honoured, n_skip would be strictly smaller.
    assert n_skip == n_default, (
        f"BF8 skip_padding_fill must be silently ignored: default cache entries={n_default}, "
        f"skip cache entries={n_skip}. A smaller skip count means fill_pad was dropped -- "
        f"that would expose logical lanes to shared-exponent corruption in BF8 sub-blocks."
    )

    # Sanity: logical values survive on both paths (PCC, since BF8 quantises).
    expected = torch_input.reshape(-1).to(torch.float32)
    assert_with_pcc(expected, ttnn.to_torch(tt_default).reshape(-1).to(torch.float32), 0.99)
    assert_with_pcc(expected, ttnn.to_torch(tt_skip).reshape(-1).to(torch.float32), 0.99)
