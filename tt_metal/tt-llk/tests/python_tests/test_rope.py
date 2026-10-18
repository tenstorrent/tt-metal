# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
SFPU RoPE test (Blackhole only). Covers experimental/ckernel_sfpu_rope.h.

sfpu_rope_all_rows rotates complex pairs in Dest. Adjacent columns contain pairs:
    x'_even = cos*x_even - sin*x_odd
    x'_odd  = sin*x_even + cos*x_odd

Only 4 rows per face are touched.

Strides:
    64: operands in their own Dest tiles, for copy_tile.
    32: operands packed two per tile, for dense matmul.
"""

import math

import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    ELEMENTS_PER_TILE,
    RopeGolden,
    get_golden_generator,
    rope_bands,
    rope_rotated_rows,
    truncate_to_bfloat16,
)
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import ROPE, TILE_COUNT

pytestmark = blackhole_only

FORMATS = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)

TILE_ROWS = 64
ROW_DATUMS = 16
MAX_DEST_TILES = 8  # 16-bit Dest
TILE_SLOT_STRIDE = 64
DENSE_STRIDE = 32


def _round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def _geometry(ht, wt, stride, operands_first=False):
    """Operand addresses in DEST rows, for `ht` heads of `wt` width tiles."""
    x_rows = ht * wt * stride
    cs_rows = wt * stride

    if operands_first:
        cos_base = 0
        x_base = _round_up(2 * cs_rows, TILE_ROWS)
    else:
        x_base = 0
        cos_base = _round_up(x_rows, TILE_ROWS)

    return {
        "ht": ht,
        "wt": wt,
        "x_base": x_base,
        "x_stride": stride,
        "cos_base": cos_base,
        "sin_base": cos_base + cs_rows,
        "cs_stride": stride,
    }


def _dest_tiles(geometry):
    """Dest tile slots the operands span."""
    last = max(
        geometry["x_base"] + geometry["ht"] * geometry["wt"] * geometry["x_stride"],
        geometry["sin_base"] + geometry["wt"] * geometry["cs_stride"],
    )
    return _round_up(last, TILE_ROWS) // TILE_ROWS


def _fits_in_dest(ht, wt, stride):
    return _dest_tiles(_geometry(ht, wt, stride)) <= MAX_DEST_TILES


def _heads(stride, wt):
    """Head counts to sweep for one (stride, wt): 1, 2, and as many as DEST holds.

    The largest fills the DEST half, running the operand addresses up to row 512 and the
    head loop out to its longest, which the ht=1 and ht=2 cases cannot do.
    """
    fitting = [
        ht
        for ht in range(1, MAX_DEST_TILES * TILE_ROWS)
        if _fits_in_dest(ht, wt, stride)
    ]
    return sorted({1, 2, fitting[-1]})


def _stimuli(geometry, tiles, seed):
    """Dest containing cos/sin over the rows read and random everywhere else."""
    generator = torch.Generator().manual_seed(seed)
    dest = torch.empty((tiles * TILE_ROWS, ROW_DATUMS), dtype=torch.float32).uniform_(
        -1.0, 1.0, generator=generator
    )

    for _, cos_row, sin_row in rope_bands(**geometry):
        for i in range(4):
            for pair in range(ROW_DATUMS // 2):
                angle = 0.11 * i + 0.29 * pair + 0.037 * cos_row
                for slot in (2 * pair, 2 * pair + 1):
                    dest[cos_row + i, slot] = math.cos(angle)
                    dest[sin_row + i, slot] = math.sin(angle)

    return dest.to(torch.bfloat16)


def _run(
    geometry,
    tiles,
    dest,
    scale_fp32=None,
    fused_cos_sin=False,
    tile_h=1,
    cos_sin_per_row=False,
    dest_acc=DestAccumulation.No,
):
    max_tiles = 4 if dest_acc == DestAccumulation.Yes else MAX_DEST_TILES
    assert tiles <= max_tiles, f"{tiles} tiles is past the Dest half"
    formats = InputOutputFormat(
        DataFormat.Float16_b,
        (
            DataFormat.Float32
            if dest_acc == DestAccumulation.Yes
            else DataFormat.Float16_b
        ),
    )

    configuration = TestConfig(
        "sources/rope_test.cpp",
        formats,
        dest_acc=dest_acc,
        templates=[
            ROPE(
                fused_cos_sin=fused_cos_sin,
                tile_h=tile_h,
                cos_sin_per_row=cos_sin_per_row,
                has_scale=scale_fp32 is not None,
                scale_fp32=0 if scale_fp32 is None else scale_fp32,
                **geometry,
            ),
        ],
        runtimes=[TILE_COUNT(tiles)],
        variant_stimuli=StimuliConfig(
            dest.flatten(),
            formats.input_format,
            torch.zeros(ELEMENTS_PER_TILE, dtype=torch.bfloat16),
            formats.input_format,
            formats.output_format,
            tile_count_A=tiles,
            tile_count_B=1,
            tile_count_res=tiles,
        ),
    )

    result = torch.tensor(
        configuration.run().result, dtype=format_dict[formats.output_format]
    )
    return result.reshape(-1, ROW_DATUMS)


def _assert_rotation(geometry, dest, device, scale=None):
    """Every Dest row matches the golden bitwise."""
    golden = get_golden_generator(RopeGolden)(dest, scale=scale, **geometry)
    rotated = rope_rotated_rows(**geometry)

    assert not torch.equal(
        device[rotated], dest[rotated].to(device.dtype)
    ), "the rotated rows came back identical to the input: the rotation did not run"

    differs = torch.nonzero((device != golden.to(device.dtype)).any(dim=1)).flatten()
    if differs.numel():
        banded = set(rotated)
        wrong = [row for row in differs.tolist() if row in banded]
        stray = [row for row in differs.tolist() if row not in banded]
        row = differs[0].item()
        report = []
        if wrong:
            report.append(f"wrong rotation at Dest rows {wrong}")
        if stray:
            report.append(f"wrote outside its 4-row bands at Dest rows {stray}")
        report.append(f"row {row}: device={device[row].tolist()}")
        report.append(f"row {row}: golden={golden[row].tolist()}")
        raise AssertionError("\n".join(report))


def _bf16_bits(value: float) -> int:
    """`value` as the fp32 bit pattern of its bf16 rounding."""
    return (
        torch.tensor([value], dtype=torch.bfloat16)
        .to(torch.float32)
        .view(torch.int32)
        .item()
    ) & 0xFFFFFFFF


@parametrize(stride=[TILE_SLOT_STRIDE, DENSE_STRIDE], wt=[1, 2], ht=_heads)
def test_rope(stride, wt, ht):
    geometry = _geometry(ht, wt, stride)
    tiles = _dest_tiles(geometry)
    dest = _stimuli(geometry, tiles, seed=101)

    device = _run(geometry, tiles, dest)

    _assert_rotation(geometry, dest, device)


# Case where x isn't placed into row 0 of Dest.
def test_rope_nonzero_x_base():
    geometry = _geometry(ht=2, wt=2, stride=TILE_SLOT_STRIDE, operands_first=True)
    assert geometry["x_base"] != 0

    tiles = _dest_tiles(geometry)
    dest = _stimuli(geometry, tiles, seed=202)

    device = _run(geometry, tiles, dest)

    _assert_rotation(geometry, dest, device)


def test_rope_scale():
    geometry = _geometry(ht=2, wt=2, stride=TILE_SLOT_STRIDE)
    tiles = _dest_tiles(geometry)
    dest = _stimuli(geometry, tiles, seed=303)

    device = _run(geometry, tiles, dest, scale_fp32=_bf16_bits(-2.0))

    _assert_rotation(geometry, dest, device, scale=-2.0)


# Has to zero out the rows.
def test_rope_zero_scale():
    geometry = _geometry(ht=2, wt=2, stride=TILE_SLOT_STRIDE)
    tiles = _dest_tiles(geometry)
    dest = _stimuli(geometry, tiles, seed=404)

    device = _run(geometry, tiles, dest, scale_fp32=_bf16_bits(0.0))

    _assert_rotation(geometry, dest, device, scale=0.0)

    rotated = rope_rotated_rows(**geometry)
    assert bool(
        (device[rotated].to(torch.float32) == 0.0).all()
    ), f"scale=0 must zero every rotated row:\n{device[rotated]}"


# A rotation by 90 degrees: cos=0, sin=1 sends (e, o) to (-o, e).
def test_rope_quarter_turn():
    geometry = _geometry(ht=1, wt=1, stride=TILE_SLOT_STRIDE)
    tiles = _dest_tiles(geometry)

    generator = torch.Generator().manual_seed(606)
    dest = torch.empty((tiles * TILE_ROWS, ROW_DATUMS), dtype=torch.float32).uniform_(
        -1.0, 1.0, generator=generator
    )
    for _, cos_row, sin_row in rope_bands(**geometry):
        dest[cos_row : cos_row + 4, :] = 0.0
        dest[sin_row : sin_row + 4, :] = 1.0
    dest = dest.to(torch.bfloat16)

    device = _run(geometry, tiles, dest)

    rotated = rope_rotated_rows(**geometry)
    even = torch.arange(0, ROW_DATUMS, 2)
    odd = even + 1
    assert torch.equal(device[rotated][:, even], -dest[rotated][:, odd]), (
        "a quarter turn must send x_even to -x_odd\n"
        f"device={device[rotated][:, even]}\nexpected={-dest[rotated][:, odd]}"
    )
    assert torch.equal(
        device[rotated][:, odd], dest[rotated][:, even]
    ), "a quarter turn must send x_odd to x_even"


def _check_fused_cos_sin(
    tile_h,
    x_stride,
    per_row,
    scale,
    ht=2,
    wt=2,
    x_base=0,
    cs_base=256,
    dest_acc=DestAccumulation.No,
):
    geometry = dict(
        ht=ht,
        wt=wt,
        x_base=x_base,
        x_stride=x_stride,
        cos_base=cs_base,
        sin_base=cs_base,
        cs_stride=64,
    )
    tiles = _dest_tiles(geometry)
    generator = torch.Generator().manual_seed(707)
    dest = (
        torch.empty((tiles * TILE_ROWS, ROW_DATUMS))
        .uniform_(-1.0, 1.0, generator=generator)
        .to(torch.bfloat16)
    )
    # Independent phase for each lane/row distinguishes a per-row load from a
    # reused decode phase. Distinct heads expose an incorrect x head stride.
    for w in range(wt):
        for row in range(TILE_ROWS):
            for pair in range(ROW_DATUMS // 2):
                angle = 0.07 * row + 0.19 * pair + 0.3 * w
                dest[cs_base + 64 * w + row, 2 * pair] = math.cos(angle)
                dest[cs_base + 64 * w + row, 2 * pair + 1] = math.sin(angle)
    output_dtype = torch.float32 if dest_acc == DestAccumulation.Yes else torch.bfloat16
    golden = dest.to(output_dtype).clone()
    effective_scale = 1.0 if scale is None else scale
    for h in range(ht):
        for w in range(wt):
            for logical_row in range(((tile_h + 3) // 4) * 4):
                for face in range(2):
                    row = (logical_row // 16) * 32 + face * 16 + logical_row % 16
                    phase_row = row if per_row else face * 16 + logical_row % 4
                    phase = dest[cs_base + 64 * w + phase_row].float()
                    x_row = x_base + x_stride * (h * wt + w) + row
                    values = dest[x_row].float()
                    cos = phase[0::2] * effective_scale
                    sin = phase[1::2] * effective_scale
                    even = cos * values[0::2] - sin * values[1::2]
                    odd = sin * values[0::2] + cos * values[1::2]
                    if dest_acc == DestAccumulation.No:
                        even, odd = truncate_to_bfloat16(even), truncate_to_bfloat16(
                            odd
                        )
                    golden[x_row, 0::2], golden[x_row, 1::2] = even, odd
    device = _run(
        geometry,
        tiles,
        dest,
        scale_fp32=None if scale is None else _bf16_bits(scale),
        fused_cos_sin=True,
        tile_h=tile_h,
        cos_sin_per_row=per_row,
        dest_acc=dest_acc,
    )
    # BF16 SFPSTORE truncates; FP32 SFPSTORE preserves the LREG result. Compare
    # every row, including phase tiles and untouched padding in partial-height tiles.
    assert torch.equal(device, golden)


@parametrize(
    tile_h=[1, 2, 4, 8, 16, 32],
    # Dense matmul slots hold the top two faces; 32 live rows need a full tile slot.
    x_stride=lambda tile_h: (
        [TILE_SLOT_STRIDE, DENSE_STRIDE] if tile_h <= 16 else [TILE_SLOT_STRIDE]
    ),
    per_row=[False, True],
    scale=[None, 0.0, -2.0],
)
def test_rope_fused_cos_sin(tile_h, x_stride, per_row, scale):
    """Fused phases for copy-tile and dense matmul layouts, including untouched rows."""
    _check_fused_cos_sin(tile_h, x_stride, per_row, scale)


@parametrize(
    # (heads, width tiles, x stride, height, per-row phase, scale, phases first)
    case=[
        (2, 1, TILE_SLOT_STRIDE, 1, False, None, False),
        (1, 2, TILE_SLOT_STRIDE, 2, True, -2.0, False),
        (2, 1, DENSE_STRIDE, 4, False, -2.0, False),
        (1, 2, DENSE_STRIDE, 8, True, None, False),
        (1, 2, TILE_SLOT_STRIDE, 16, False, 0.0, False),
        (1, 2, TILE_SLOT_STRIDE, 32, True, None, False),
        (1, 1, TILE_SLOT_STRIDE, 32, False, -2.0, True),
        (1, 2, TILE_SLOT_STRIDE, 32, True, 0.0, False),
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_rope_fused_dest_precision(case, dest_acc):
    """BF16 inputs unpacked into either DEST representation, within its half capacity."""
    ht, wt, x_stride, tile_h, per_row, scale, phases_first = case
    x_base = wt * TILE_ROWS if phases_first else 0
    cs_base = 0 if phases_first else _round_up(ht * wt * x_stride, TILE_ROWS)
    _check_fused_cos_sin(
        tile_h,
        x_stride,
        per_row,
        scale,
        ht=ht,
        wt=wt,
        x_base=x_base,
        cs_base=cs_base,
        dest_acc=dest_acc,
    )
