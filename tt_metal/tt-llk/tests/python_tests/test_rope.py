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
)
from helpers.llk_params import format_dict
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


def _run(geometry, tiles, dest, scale_fp32=None):
    assert tiles <= MAX_DEST_TILES, f"{tiles} tiles is past the Dest half"

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
