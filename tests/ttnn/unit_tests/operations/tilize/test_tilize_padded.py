# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 — padding: `pad_mode`, `pad_value`, `alignment`, ranks 0 and 1.

Three things are pinned, and the third is the one the values alone cannot show:

  1. THE PAD REGION holds exactly the fill. Only the PADDED readback exposes it
     (`to_torch_with_padded_shape`); `to_torch` strips it, so a test that
     compared the logical view alone would pass on a kernel that filled the pad
     with stale L1.
  2. THE LOGICAL SHAPE DOES NOT GROW. Promoting it is the named bug: the output
     of a padded tilize is still the caller's tensor, just physically larger.
  3. THE FILL IS PRODUCED IN THE KERNEL, and only where a pad region exists.
     `plan.pad_active` is asserted directly, because a host-side pre-fill or an
     always-on fill path would both pass (1) and (2) while changing what the op
     is. The False direction matters just as much: an already-aligned call must
     stay on the Phase 0 reader branch even when `pad_value=` is passed.

tilize does no arithmetic, so every value assertion here is `torch.equal`.
Device comes from the directory conftest's module-scoped `device` fixture.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from ttnn.operations.tilize import tilize
from ttnn.operations.tilize.tilize_program_descriptor import (
    CB_PAD_ROW,
    create_program_descriptor,
    derive_plan,
    pad_fill_word,
)

_DRAM = ttnn.DRAM_MEMORY_CONFIG
_L1 = ttnn.L1_MEMORY_CONFIG


def _crs(end):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(*end))})


def _height_sharded(shard_shape, end):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_crs(end), shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )


def _nd_sharded(shard_shape, end):
    return ttnn.MemoryConfig(
        ttnn.BufferType.L1,
        ttnn.NdShardSpec(ttnn.Shape(list(shard_shape)), _crs(end), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _pad_expected(x, padded_shape, pad_value):
    """`F.pad` out to `padded_shape`, left-expanding a sub-rank input with 1s —
    the same oracle the golden helpers use, restated so this file stands alone."""
    if x.dim() < len(padded_shape):
        x = x.reshape((1,) * (len(padded_shape) - x.dim()) + tuple(x.shape))
    pads = tuple(j for i in reversed(range(x.dim())) for j in (0, padded_shape[i] - x.shape[i]))
    return torch.nn.functional.pad(x, pads, value=pad_value)


def _run(device, shape, pad_value, *, target=None, in_mc=_DRAM, out_mc=_DRAM, low_l1=False, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(shape, dtype=torch.bfloat16) if shape else torch.tensor(-2.5, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mc)
    kwargs = {"pad_value": pad_value}
    if target is not None:
        kwargs["output_padded_shape"] = target
    if low_l1:
        kwargs["low_l1"] = True
    out = tilize(tt_in, out_mc, dtype=ttnn.bfloat16, **kwargs)

    # (2) the logical view is untouched — shape AND values.
    logical = ttnn.to_torch(out)
    assert list(logical.shape) == list(shape), f"logical shape grew: {list(logical.shape)} != {list(shape)}"
    assert torch.equal(logical, x), "logical region is not the input"

    # (1) every padded position holds exactly the fill.
    padded = out.cpu().to_torch_with_padded_shape()
    if target is not None:
        assert list(padded.shape) == list(target), f"padded shape {list(padded.shape)} != requested {target}"
    expected = _pad_expected(x, list(padded.shape), pad_value)
    assert torch.equal(padded, expected), f"{int((padded != expected).sum())} of {expected.numel()} positions differ"
    return out


# ---------------------------------------------------------------------------
# 1 + 2. The pad region and the logical view, across every alignment and sign
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,pad_value",
    [
        ((1, 1, 32, 50), 0.0),  # w_non_aligned, zero fill
        ((1, 1, 50, 64), 3.0),  # h_non_aligned, positive fill
        ((1, 1, 50, 50), -7.0),  # hw_non_aligned, negative fill
        ((1, 1, 30, 32), 0.5),  # sub-tile in H, a fractional fill
        ((50, 50), 0.0),  # rank 2
        ((3, 50, 64), 2.0),  # rank 3: the H tail per image
        ((1, 2, 1, 50, 50), -1.0),  # rank 5: the leading-dim fold survives the pad
        ((64,), 0.0),  # rank 1: the pad SYNTHESIZES both tile dims
        ((), 0.0),  # rank 0: a scalar becomes one [32,32] tile
    ],
    ids=["w_tail", "h_tail", "hw_tails", "subtile_h", "rank2", "rank3", "rank5", "rank1", "rank0"],
)
def test_pad_auto(device, shape, pad_value):
    """`pad_mode="auto"`: the target is the last two dims rounded to the tile."""
    _run(device, shape, pad_value)


@pytest.mark.parametrize(
    "shape,target,pad_value",
    [
        ((1, 1, 32, 50), [1, 1, 32, 128], 0.0),  # W past the round: whole pad tile-COLUMNS
        ((1, 1, 50, 50), [1, 1, 128, 128], 5.0),  # both dims past the round
        ((1, 1, 32, 64), [1, 1, 64, 128], 0.0),  # aligned input, whole pad TILES only
        ((1, 1, 30, 32), [1, 1, 32, 32], -4.0),  # explicit AT the round == auto
        ((3, 50, 96), [3, 64, 96], 10.2),  # rank 3, explicit at the round
    ],
    ids=["beyond_round_w", "beyond_round_hw", "whole_pad_tiles", "exact_round", "rank3_at_round"],
)
def test_pad_explicit(device, shape, target, pad_value):
    """`pad_mode="explicit"`: a caller-named target, which MAY exceed the round.
    Past the round the pad tiles have no input bytes at all."""
    _run(device, shape, pad_value, target=target)


@pytest.mark.parametrize(
    "label,kwargs",
    [
        ("l1_to_l1", {"in_mc": _L1, "out_mc": _L1}),
        ("dram_to_l1", {"out_mc": _L1}),
        ("l1_to_dram", {"in_mc": _L1}),
        ("low_l1", {"low_l1": True}),
    ],
)
def test_pad_crossed_with_placement(device, label, kwargs):
    """The fill has to land whatever the placement and however tight the budget."""
    _run(device, (1, 1, 50, 50), 0.0, **kwargs)


def test_pad_into_height_sharded_output(device):
    """The fill lands in a SHARDED output too: the packer writes whole tiles,
    pad positions included, straight into the output shard."""
    _run(device, (1, 1, 50, 64), 0.0, out_mc=_height_sharded((32, 64), (1, 0)))


def test_pad_from_nd_sharded_input(device):
    """An L1-sharded INPUT under a pad. The shard holds only the caller's own
    bytes, so this call reads through the accessor rather than zero-copy — and
    the shard cuts the row width, so the reader is on its strided page walk with
    a W tail on top of it."""
    _run(device, (3, 100, 158), 10.2, target=[3, 128, 160], in_mc=_nd_sharded([2, 64, 96], (1, 0)))


def test_pad_into_nd_sharded_output(device):
    _run(device, (3, 50, 96), 10.2, target=[3, 64, 96], out_mc=_nd_sharded([1, 64, 96], (1, 0)))


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 1, 2048), (1, 1, 32, 4090), (8, 1, 249, 2048)],
    ids=["single_stick", "short_wide_w_tail", "leading_fold_h_tail"],
)
def test_pad_at_grid_scale(device, shape):
    """Padded WORK GEOMETRY. `[1,1,1,2048]` is 31 pad rows per tile-row on a
    width-split grid; `[8,1,249,2048]` reaches 64 tile-rows THROUGH the fold
    with an H tail in every image, which is the case a reader that spans one
    contiguous stick run across images gets wrong."""
    _run(device, shape, 0.0)


# ---------------------------------------------------------------------------
# 3. The fill is a KERNEL path, and only where a pad region exists
# ---------------------------------------------------------------------------


def _plan_and_program(device, shape, *, pad_value, target=None):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=_DRAM)
    kwargs = {"pad_value": pad_value}
    if target is not None:
        kwargs["output_padded_shape"] = target
    out = tilize(tt_in, _DRAM, dtype=ttnn.bfloat16, **kwargs)
    grid = device.compute_with_storage_grid_size()
    plan = derive_plan(tt_in, out, low_l1=False, grid=grid, pad_value=pad_value)
    return plan, create_program_descriptor(tt_in, out, pad_value=pad_value)


@pytest.mark.parametrize(
    "shape,target,expect_active",
    [
        ((1, 1, 50, 50), None, True),  # a real pad region
        ((1, 1, 64, 64), None, False),  # aligned + pad_value= -> nothing to fill
        ((1, 1, 64, 64), [1, 1, 64, 64], False),  # explicit AT the shape -> nothing to fill
        ((1, 1, 64, 64), [1, 1, 128, 64], True),  # explicit past it -> whole pad tile-rows
    ],
    ids=["tails", "aligned_no_region", "explicit_no_region", "explicit_whole_tiles"],
)
def test_pad_active_tracks_the_region_not_the_argument(device, shape, target, expect_active):
    """`pad_active` is derived from the GEOMETRY, which is what keeps an
    already-aligned call — padded argument or not — on the Phase 0 reader
    branch, byte-identical to before this refinement."""
    plan, program = _plan_and_program(device, shape, pad_value=1.0, target=target)
    assert plan.pad_active is expect_active
    # cb_pad_row exists iff the fill path does, and costs exactly one block row.
    pad_cbs = [cb for cb in program.cbs if cb.format_descriptors[0].buffer_index == CB_PAD_ROW]
    assert len(pad_cbs) == (1 if expect_active else 0)
    if expect_active:
        assert pad_cbs[0].total_size == plan.block_row_bytes


def test_pad_row_scratch_is_bounded_by_the_block_not_the_tensor():
    """cb_pad_row is ONE block row, so it scales with `block_width_tiles` and
    with no tensor dimension — the same bound as every other CB in this op."""
    assert pad_fill_word(ttnn.bfloat16, 0.0, 2) == 0x0000
    assert pad_fill_word(ttnn.bfloat16, 1.0, 2) == 0x3F80
    assert pad_fill_word(ttnn.bfloat16, -1.0, 2) == 0xBF80
    # A negative integer fill is a two's-complement bit_cast at the ELEMENT
    # width and must not truncate — the rule Refinement 5's integer dtypes rely on.
    assert pad_fill_word(ttnn.int32, -1, 4) == 0xFFFFFFFF
    assert pad_fill_word(ttnn.uint16, -1, 2) == 0xFFFF
    assert pad_fill_word(ttnn.uint8, -1, 1) == 0xFF
    # bfloat16 ROUNDS (to nearest even), not truncates. Checked against torch's
    # own conversion, because the pad oracle is
    # `F.pad(x.bfloat16(), value=v)` — a truncating encoder puts a different
    # number in the pad region than the oracle for any fill that is not exactly
    # representable, and 0.1 / 1.00390625 / 1.01171875 are the three shapes of
    # that: round up, an exact tie down to the even mantissa, and a tie up.
    for value in (0.1, -0.1, 1.00390625, 1.01171875, 3.4e38, -7.0, 1e-40):
        torch_bits = int(torch.tensor([value], dtype=torch.bfloat16).view(torch.int16).item() & 0xFFFF)
        assert pad_fill_word(ttnn.bfloat16, value, 2) == torch_bits, f"bf16 encoding of {value}"


# ---------------------------------------------------------------------------
# Malformed padding requests stay ValueErrors, not support refusals
# ---------------------------------------------------------------------------


def test_non_aligned_without_padding_is_refused(device, expect_error):
    """Padding is opt-in: an unaligned input with no padding argument is a
    malformed REQUEST, not a silent pad."""
    torch.manual_seed(0)
    x = torch.randn((1, 1, 50, 50), dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=_DRAM)
    with expect_error(ValueError, "not tile-aligned"):
        tilize(tt_in, _DRAM)


@pytest.mark.parametrize(
    "target,match",
    [
        ([1, 1, 32, 32], "smaller than the input"),
        ([1, 1, 64, 50], "whole number of"),
    ],
    ids=["target_too_small", "target_not_tile_shaped"],
)
def test_malformed_explicit_target(device, target, match, expect_error):
    torch.manual_seed(0)
    x = torch.randn((1, 1, 50, 50), dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=_DRAM)
    with expect_error(ValueError, match):
        tilize(tt_in, _DRAM, output_padded_shape=target, pad_value=0.0)
