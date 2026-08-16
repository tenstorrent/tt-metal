# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
SFPU RoPE test (Blackhole only).

Covers tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_rope.h
(`sfpu_rope_configure_addrmod` / `sfpu_rope_dest_setup` / `sfpu_rope_all_rows`).

`sfpu_rope_all_rows` rotates complex pairs in place in DEST. It takes its operands as
absolute DEST rows rather than tile indices, so everything here is expressed in DEST
rows: a tile slot is 64 rows, a face is 16, and one DEST row is the 16 datums of one
face row. That makes the L1 tile a [64, 16] view with no tilize step -- the packed
face-major order of a tile *is* its DEST rows.

    x'_even = cos*x_even - sin*x_odd
    x'_odd  = sin*x_even + cos*x_odd

Only 4 rows per face are touched: one SFPU vector is 4 rows x 16 columns of both
parities, and the LLK issues exactly one per (width tile, face). Everything else in
DEST, including the cos/sin operands, must come back bit-identical, so every tile is
packed back out and checked.

cos/sin layout: one angle per complex pair, duplicated across both slots of the pair.
A single even-parity load then serves both x parities, which is what the stimuli here
build and what the golden reads (the even slot).

Strides. Both layouts the LLK documents are swept:
  * 64 -- operands in their own DEST tile slots, the copy_tile shape.
  * 32 -- operands packed two per slot, the dense-packed matmul shape, where head h
    lands on faces 0/1 of a slot and head h+1 on faces 2/3.
The stride only enters through address arithmetic, so getting it wrong reads or writes
a neighbouring head. Sweeping both is the point of driving the LLK directly rather
than through the compute API, which only ever passes 64.

Formats. The loads and stores are hardcoded to InstrModLoadStore::FP16B, so this op is
bf16-DEST only and dest_acc is not an axis: with a 32-bit DEST the same addresses would
reinterpret fp32 words as bf16 pairs. The compute-API wrapper does not gate on
DST_ACCUM_MODE, so a caller built with fp32 accumulation would hit that.
"""

import math

import pytest
import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    ELEMENTS_PER_TILE,
    RopeGolden,
    get_golden_generator,
)
from helpers.llk_params import format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import ROPE, TILE_COUNT
from helpers.utils import passed_test

pytestmark = blackhole_only

FORMATS = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)

TILE_ROWS = 64  # DEST rows per tile slot
FACE_ROWS = 16  # DEST rows per face
ROW_DATUMS = 16  # datums per DEST row (one face row)
VECTOR_ROWS = 4  # DEST rows one SFPU vector covers
MAX_DEST_TILES = 8  # tile slots in half of a 16-bit DEST

# Strides in DEST rows: the copy_tile layout and the dense-packed matmul layout.
TILE_SLOT_STRIDE = 64
DENSE_STRIDE = 32


def _round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def _geometry(ht, wt, stride):
    """Operand addresses in DEST rows, for `ht` heads of `wt` width tiles.

    x starts at row 0 and cos/sin follow on the next tile boundary, so that the number
    of DEST tiles the kernel has to unpack stays minimal for both strides.
    """
    x_rows = ht * wt * stride
    cos_base = _round_up(x_rows, TILE_ROWS)
    return {
        "ht": ht,
        "wt": wt,
        "x_base": 0,
        "x_stride": stride,
        "cos_base": cos_base,
        "sin_base": cos_base + wt * stride,
        "cs_stride": stride,
    }


def _dest_tiles(geometry):
    """DEST tile slots the operands span."""
    last = geometry["sin_base"] + geometry["wt"] * geometry["cs_stride"]
    return _round_up(last, TILE_ROWS) // TILE_ROWS


def _fits_in_dest(ht, wt, stride):
    return _dest_tiles(_geometry(ht, wt, stride)) <= MAX_DEST_TILES


def _stimuli(geometry, tiles, seed):
    """DEST as [tiles * 64, 16] bf16: random everywhere, cos/sin over the rows read.

    Filling the whole of DEST with data rather than zeros is deliberate. A stray write
    outside the rotated bands then shows up as a mismatch instead of landing on a value
    that was already zero, and the padding rows of the cos/sin tiles are not special.
    """
    generator = torch.Generator().manual_seed(seed)
    dest = torch.empty((tiles * TILE_ROWS, ROW_DATUMS), dtype=torch.float32).uniform_(
        -1.0, 1.0, generator=generator
    )

    # RopeGolden's geometry helpers are used through the class, not through
    # get_golden_generator: compile-producer swaps the instance for a dummy.
    for _, cos_row, sin_row in RopeGolden.bands(**geometry):
        for i in range(VECTOR_ROWS):
            for pair in range(ROW_DATUMS // 2):
                # Keyed on the absolute cos row, so every (width tile, face, row) pair
                # gets its own angle and reading the wrong one changes the result. sin
                # keys on the same row, since a pair's cos and sin share an angle.
                angle = 0.11 * i + 0.29 * pair + 0.037 * cos_row
                for slot in (2 * pair, 2 * pair + 1):
                    dest[cos_row + i, slot] = math.cos(angle)
                    dest[sin_row + i, slot] = math.sin(angle)

    return dest.to(torch.bfloat16)


def _run(geometry, tiles, dest, scale_fp32=None):
    """Rotate `dest` on device and return the packed result as [tiles * 64, 16]."""
    configuration = TestConfig(
        "sources/rope_test.cpp",
        FORMATS,
        templates=[
            ROPE(
                has_scale=scale_fp32 is not None,
                scale_fp32=0 if scale_fp32 is None else scale_fp32,
                **geometry,
            ),
        ],
        runtimes=[TILE_COUNT(tiles)],
        variant_stimuli=StimuliConfig(
            dest.flatten(),
            FORMATS.input_format,
            torch.zeros(ELEMENTS_PER_TILE, dtype=torch.bfloat16),
            FORMATS.input_format,
            FORMATS.output_format,
            tile_count_A=tiles,
            tile_count_B=1,
            tile_count_res=tiles,
        ),
    )

    result = torch.tensor(
        configuration.run().result, dtype=format_dict[FORMATS.output_format]
    )
    return result.reshape(-1, ROW_DATUMS)


def _assert_rotation(geometry, dest, device, scale=None):
    """The rotated bands match the golden; every other DEST row is untouched."""
    golden_generator = get_golden_generator(RopeGolden)
    golden = golden_generator(dest, scale=scale, **geometry)

    rotated = RopeGolden.rotated_rows(**geometry)
    assert passed_test(
        golden[rotated].flatten(),
        device[rotated].flatten(),
        FORMATS.output_format,
        print_errors=True,
    ), f"rotation does not match golden (rows {rotated[0]}..{rotated[-1]})"

    # Everything the op does not write goes unpack -> DEST -> pack with no arithmetic,
    # so it has to come back bit-identical, cos/sin operands included. An off-by-one in
    # the stride or the face offset lands here rather than in the check above.
    untouched = torch.ones(dest.shape[0], dtype=torch.bool)
    untouched[rotated] = False
    differs = (device != golden.to(device.dtype)).any(dim=1)
    stray = torch.nonzero(untouched & differs).flatten()
    assert (
        stray.numel() == 0
    ), f"rope wrote outside its 4-row bands, at DEST rows {stray.tolist()}"


def _bf16_bits(value: float) -> int:
    """`value` as the fp32 bit pattern of its bf16 rounding.

    The scale reaches the LLK as an fp32 pattern it loads with two SFPLOADIs, but the
    stimuli and the golden carry bf16, so keeping the scale on the bf16 lattice makes
    the comparison exact in the scale itself.
    """
    return (
        torch.tensor([value], dtype=torch.bfloat16)
        .to(torch.float32)
        .view(torch.int32)
        .item()
    ) & 0xFFFFFFFF


# Heads and width tiles across both strides. Wt=2 is head_dim 64, where each width tile
# has its own cos/sin operand, so a variant that reads only the first one fails.
@parametrize(
    stride=[TILE_SLOT_STRIDE, DENSE_STRIDE],
    ht=[1, 2, 4],
    wt=[1, 2],
)
def test_rope(stride, ht, wt):
    if not _fits_in_dest(ht, wt, stride):
        pytest.skip(f"ht={ht} wt={wt} stride={stride} needs more than half of DEST")

    geometry = _geometry(ht, wt, stride)
    tiles = _dest_tiles(geometry)
    dest = _stimuli(geometry, tiles, seed=101)

    device = _run(geometry, tiles, dest)

    _assert_rotation(geometry, dest, device)


# The heads that fill half of DEST at stride 64: 6 x tiles plus one cos and one sin.
def test_rope_fills_dest():
    geometry = _geometry(ht=6, wt=1, stride=TILE_SLOT_STRIDE)
    tiles = _dest_tiles(geometry)
    assert tiles == MAX_DEST_TILES

    dest = _stimuli(geometry, tiles, seed=202)

    device = _run(geometry, tiles, dest)

    _assert_rotation(geometry, dest, device)


# has_scale folds a deferred normalization into cos/sin, once per (width tile, face)
# rather than per head. scale=0 is the case the promotion made explicit by dropping the
# `scale_fp32 = 0` default: it must zero the rotated rows, not pass them through.
@parametrize(scale=[0.0, 0.5, 1.0, 1.75, -2.0], wt=[1, 2])
def test_rope_scale(scale, wt):
    geometry = _geometry(ht=2, wt=wt, stride=TILE_SLOT_STRIDE)
    tiles = _dest_tiles(geometry)
    dest = _stimuli(geometry, tiles, seed=303)

    device = _run(geometry, tiles, dest, scale_fp32=_bf16_bits(scale))

    _assert_rotation(geometry, dest, device, scale=scale)

    if scale == 0.0:
        rotated = RopeGolden.rotated_rows(**geometry)
        assert bool(
            (device[rotated].to(torch.float32) == 0.0).all()
        ), f"scale=0 must zero every rotated row:\n{device[rotated]}"


# A rotation by 90 degrees is exact in bf16: cos=0, sin=1 sends (e, o) to (-o, e). This
# pins the sign convention, which a tolerance check against a golden that shares the
# convention cannot catch.
def test_rope_quarter_turn():
    geometry = _geometry(ht=1, wt=1, stride=TILE_SLOT_STRIDE)
    tiles = _dest_tiles(geometry)

    generator = torch.Generator().manual_seed(404)
    dest = torch.empty((tiles * TILE_ROWS, ROW_DATUMS), dtype=torch.float32).uniform_(
        -1.0, 1.0, generator=generator
    )
    # RopeGolden's geometry helpers are used through the class, not through
    # get_golden_generator: compile-producer swaps the instance for a dummy.
    for _, cos_row, sin_row in RopeGolden.bands(**geometry):
        dest[cos_row : cos_row + VECTOR_ROWS, :] = 0.0
        dest[sin_row : sin_row + VECTOR_ROWS, :] = 1.0
    dest = dest.to(torch.bfloat16)

    device = _run(geometry, tiles, dest)

    rotated = RopeGolden.rotated_rows(**geometry)
    even = torch.arange(0, ROW_DATUMS, 2)
    odd = even + 1
    assert torch.equal(device[rotated][:, even], -dest[rotated][:, odd]), (
        "a quarter turn must send x_even to -x_odd\n"
        f"device={device[rotated][:, even]}\nexpected={-dest[rotated][:, odd]}"
    )
    assert torch.equal(
        device[rotated][:, odd], dest[rotated][:, even]
    ), "a quarter turn must send x_odd to x_even"
