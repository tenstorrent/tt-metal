# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import re

import pytest
import torch
from conftest import skip_for_coverage
from helpers.bfp_format_utils import bfp4b_to_float16b, bfp8b_to_float16b
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, Tilize
from helpers.logger import strip_ansi
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    NUM_FACES,
    TILIZE,
    generate_input_dim,
)

pytestmark = skip_for_coverage


# Float values are always rendered with at least one fractional digit (the
# formatter uses {:7.2f}), so requiring \d+ after the dot rejects row
# prefixes like "01." as floats. The int branch handles {:7d} cells.
_NUMBER_RE = re.compile(r"-?\d+\.\d+(?:[eE][+-]?\d+)?|-?\.\d+(?:[eE][+-]?\d+)?|-?\d+")
_ANGLED_MARKER_RE = re.compile(r"<[^>]*>")
# A tile slice arrives as a single multi-line string (one DEVICE_PRINT call,
# embedded \n between rows), so MULTILINE is required to anchor at every row.
_ROW_PREFIX_RE = re.compile(r"^\d+\.\s+", re.MULTILINE)


def _extract_floats(lines: list[str]) -> list[float]:
    """Pull every number out of the given lines into a list of floats.
    Strips the dprint '[RISC|...]' prefix, tile truncation notices,
    ANSI color codes, and the tensor row prefix."""

    out: list[float] = []
    for line in lines:
        body = line.split("] ", 1)[-1]
        body = _ANGLED_MARKER_RE.sub("", body)
        body = strip_ansi(body)
        body = _ROW_PREFIX_RE.sub("", body)
        out.extend(float(m) for m in _NUMBER_RE.findall(body))
    return out


def _records(lines: list[str]) -> list[list[str]]:
    """Group device_print_lines into per-DEVICE_PRINT chunks.
    A new chunk starts at every '[RISC|file:line]' header.
    Multi-line records attach to the preceding chunk."""

    chunks: list[list[str]] = []
    for line in lines:
        if line.startswith("["):
            chunks.append([line])
        elif chunks:
            chunks[-1].append(line)
    return chunks


def test_device_print():
    formats = input_output_formats([DataFormat.Int32])[0]

    configuration = TestConfig(
        "sources/device_print_test.cpp",
        formats,
        dest_acc=DestAccumulation.Yes,
        requires_device_print=True,  # Required for this test to pass in CI.
    )
    outcome = configuration.run()
    lines = outcome.device_print_lines

    full = "".join(lines)
    assert full, "No device print output received"

    # Unpack: multi-size args (reordering takes place)
    assert "unpack: i8=-1 u8=255 i16=-100 u16=65535" in full
    assert "_unpack" in full

    # Enum resolution from debug info:
    #   regular enum -> single name
    #   flag enum -> "name1 | name2"
    #   '#' spec -> fully qualified "type::name | type::name"
    #   unknown value -> "(type)value"
    assert "unpack: enum=Green" in full
    assert "unpack: flag=R | X" in full
    assert "unpack: flag_full=Perm::R | Perm::W" in full
    assert "unpack: flag_unk=(Perm)24" in full

    # Name resolution follows DWARF declaration order, not numeric order.
    # Rev is declared { Z=4, Y=2, X=1 }, so 7 -> "Z | Y | X", not "X | Y | Z".
    assert "unpack: flag_rev=Z | Y | X" in full

    array_lines = [line for line in lines if "unpack: array=" in line]
    assert array_lines, "unpack: array= line missing"
    assert _extract_floats(array_lines) == pytest.approx(
        [1.0, 2.0, 3.0, 4.0]
    )  # Rounding

    # Math: one print per type category
    assert "math: i32=-1 u32=65536" in full
    assert "math: float=1.0" in full
    assert "math: bool=true false" in full
    assert "math: ptr=0xdeadbeef" in full
    assert "_math" in full  # CTSTR
    assert "math: hex=00000abc" in full
    assert "math: pad=    test" in full

    # We print 2048 iterations (weighing 8 bytes each) to force a drain.
    # Whether this test hits the stall path depends on the buffer size.
    missing = [i for i in range(2048) if f"w={i}" not in full]
    assert not missing, (
        f"Missing {len(missing)} of 2048 wrap iterations; "
        f"first 10 missing: {missing[:10]}"
    )

    # Pack
    assert "pack: i64=-1000000" in full
    assert "_pack" in full

    # SFPU is only built on Quasar
    if TestConfig.CHIP_ARCH == ChipArchitecture.QUASAR:
        assert "sfpu: u8=3 i8=-1" in full
        assert "_sfpu" in full


# Mirror SliceRange::{hw0_32_8,hw0_32_4} in dprint_tile_test.cpp:
# the 4x4 slice that always fits and the 8x8 one that gets truncated.
_TILE_SLICE_INDICES = tuple(range(0, 32, 8))
_TRUNC_SLICE_INDICES = tuple(range(0, 32, 4))

# Bytes per element in tile_slice's L1 walk. Bfp* formats emit one (exp, mantissa)
# byte pair per element, so they count as 2 even though logical element size differs.
_BYTES_PER_ELT = {
    DataFormat.Int8: 1,
    DataFormat.UInt8: 1,
    DataFormat.Float16_b: 2,
    DataFormat.UInt16: 2,
    DataFormat.Bfp4_b: 2,
    DataFormat.Bfp8_b: 2,
    DataFormat.Float32: 4,
    DataFormat.Int32: 4,
    DataFormat.UInt32: 4,
}


def _tilized_index(h: int, w: int) -> int:
    face_r, face_c = h // 16, w // 16
    return (face_r * 2 + face_c) * 256 + (h % 16) * 16 + (w % 16)


# Formats test_dprint_tile exercises. It requires SrcA data format support.
# WH/BH accept everything in _BYTES_PER_ELT.
_TILE_TEST_FORMATS = (
    [
        DataFormat.Int8,
        DataFormat.UInt8,
        DataFormat.Float16_b,
        DataFormat.Float32,
        DataFormat.Int32,
    ]
    if get_chip_architecture() == ChipArchitecture.QUASAR
    else list(_BYTES_PER_ELT.keys())
)


@parametrize(
    formats=input_output_formats(_TILE_TEST_FORMATS, same=True),
)
def test_dprint_tile(formats):
    formats = formats[0]
    dest_acc = (
        DestAccumulation.Yes
        if formats.input_format.is_32_bit()
        else DestAccumulation.No
    )
    src_A, tile_cnt_A, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=[32, 32],
        stimuli_format_B=formats.input_format,
        input_dimensions_B=[32, 32],
    )
    outcome = TestConfig(
        "sources/dprint_tile_test.cpp",
        formats,
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_A,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        requires_device_print=True,
    ).run()
    records = _records(outcome.device_print_lines)

    if formats.input_format == DataFormat.Bfp8_b:
        reference = bfp8b_to_float16b(src_A)
    elif formats.input_format == DataFormat.Bfp4_b:
        reference = bfp4b_to_float16b(src_A)
    else:
        reference = src_A

    # records[0]: hw0_32_8 has 16 cells, so it always fits in 64B.
    expected_full = [
        float(reference[_tilized_index(h, w)])
        for h in _TILE_SLICE_INDICES
        for w in _TILE_SLICE_INDICES
    ]

    assert _extract_floats(records[0]) == pytest.approx(
        expected_full, abs=0.01
    )  # Rounding

    # records[1]: hw0_32_4 has 64 cells. Truncates at MAX_BYTES / bytes_per_elt;
    # for 1-byte formats the 64 cells exactly fit and no marker is emitted.
    fit_cells = 64 // _BYTES_PER_ELT[formats.input_format]
    all_64 = [
        float(reference[_tilized_index(h, w)])
        for h in _TRUNC_SLICE_INDICES
        for w in _TRUNC_SLICE_INDICES
    ]
    assert _extract_floats(records[1]) == pytest.approx(all_64[:fit_cells], abs=0.01)
    truncated = "TileSlice truncated" in "".join(records[1])
    assert truncated == (
        fit_cells < 64
    ), f"Expected truncated={fit_cells < 64} for {formats.input_format}, got {truncated}"


@parametrize(
    case=[
        (DataFormat.Float16_b, DestAccumulation.Yes, False),
        (DataFormat.Float16_b, DestAccumulation.No, False),
        (DataFormat.Float16, DestAccumulation.No, False),
    ],
)
def test_dprint_tensix(case):
    in_format, dest_acc, unpack_to_dest = case[0]

    if get_chip_architecture() == ChipArchitecture.QUASAR:
        pytest.skip("dprint_tensix_dest_reg is unsupported on Quasar")

    formats = InputOutputFormat(in_format, in_format)
    src_A, _, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=[32, 32],
        stimuli_format_B=formats.input_format,
        input_dimensions_B=[32, 32],
    )

    outcome = TestConfig(
        "sources/dprint_tensix_test.cpp",
        formats,
        templates=[generate_input_dim([32, 32], [32, 32]), TILIZE(Tilize.No)],
        runtimes=[DEST_INDEX(0), NUM_FACES(4)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_A,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
            num_faces=4,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=unpack_to_dest,
        requires_device_print=True,
    ).run()

    # Skip the "Tile ID = 0" header (its trailing "0" parses as a number) and the
    # Wormhole "WARNING: Float32 (...)" line.
    decoded = _extract_floats(
        [
            line
            for line in outcome.device_print_lines
            if "Tile ID" not in line and "WARNING" not in line
        ]
    )

    # bf16 round-trips through DEST: DestAcc.Yes widens to fp32 with zero low
    # bits; DestAcc.No keeps it as Float16_b.
    expected = src_A.to(torch.float32).flatten().tolist()
    assert decoded == pytest.approx(expected, abs=0.01)
