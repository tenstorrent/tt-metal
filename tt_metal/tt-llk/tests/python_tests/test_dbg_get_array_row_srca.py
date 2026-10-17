# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
``dbg_get_array_row(dbg_array_id::SRCA, ...)`` borrows a dest row to stage the SrcA row it
cannot read directly, and must put that row back before returning. Two tests, one per contract:

* ``test_srca_row_dump_restores_dest_row_0`` (Blackhole, which has the RISC-V window into DEST):
  the row is 32 datums wide and the SFPU saves it in two halves, so it needs two registers;
  sharing one loses the first half and the restore writes the surviving half over both. The
  kernel writes a known tile into DEST, calls the helper, and reads DEST back -- the tile must
  come back bit-exact with either DEST width. Row 0 carries planted patterns a converting save
  would change: integer values that read as float denormals, and float denormals and negative
  zero, which a flush or a half-width save alters.

* ``test_srca_row_dump_reads_srca_row`` (both architectures): the dump must be the SrcA row
  wherever the math thread's dest offset sits. The kernel unpacks a tile into SrcA, datacopies
  it into a dest tile (so the offset is where a real op leaves it: tile 0, tile 2, or the upper
  half), dumps SrcA row 0 and hands back the raw debug-bus words. The datacopy consumed the
  faces in order, so the bank the dump reads holds the last face: the words must be row 0 of
  face 3 in the dest BF16 bit layout (the host builds them from the tile it wrote). The helper
  stages and restores through the thread's offset but reads back a physical row, so without
  pinning the offset first a nonzero offset returns a stale row. The same variants check that
  the dump hands the offset back as it found it, since ops that run on the inherited offset
  would otherwise be retargeted.

Every in-tree caller passes ``dbg_array_id::DEST``, which never enters this path, so without
these tests the SrcA path can regress undetected.
"""

import struct

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DBG_SRCA_DUMP
from helpers.tile_constants import (
    FACE_C_DIM,
    MAX_FACE_ELEMENTS,
    MAX_NUM_FACES,
    MAX_TILE_ELEMENTS,
)

# Int32 bit patterns a converting save would change: an FP32-mode save flushes the ones that read as
# denormals (exponent 0) and a BF16-mode save drops their low half. Planted in dest row 0 so the
# check does not depend on the random stimuli happening to include one.
INT32_AT_RISK = [
    1,
    0x7FFFFF,
    0x00373F0B,
    -1,
    -0x7FFFFFFF,
    0x12345678,
    0x00008000,
    0x7FFFFFFF,
]
# Float bit patterns with the same property: denormals (a converting save flushes them), negative zero
# (a float compare cannot tell it from zero, so the check below is bitwise), the smallest normal and
# the largest finite value as controls.
FP32_AT_RISK_BITS = [
    0x00000001,
    0x007FFFFF,
    0x80000001,
    0x80000000,
    0x00800000,
    0x7F7FFFFF,
]
BF16_AT_RISK_BITS = [0x0001, 0x007F, 0x8001, 0x8000, 0x0080, 0x7F7F]

INPUT_DIMENSIONS = [32, 32]
# Result layout the kernel writes, shared through the template parameter.
WORDS_PER_ROW = DBG_SRCA_DUMP.WORDS_PER_ROW
DEST_OFFSET_WORD = DBG_SRCA_DUMP.DEST_OFFSET_WORD
RESULT_TILE_DEST = DBG_SRCA_DUMP.RESULT_TILE_DEST

# BF16 fields, as test_topk.py names them.
_BF16_SIGN_MASK = 0x8000
_BF16_EXP_MASK = 0x7F80
_BF16_MANT_MASK = 0x007F
_BF16_EXP_SHIFT = 7
_BF16_MANT_BITS = 8

# Where the datacopy leaves the math thread's dest offset before the dump: (dest tile, upper half).
DEST_OFFSETS = {
    "tile_0": (0, False),
    "tile_2": (2, False),
    "upper_half": (0, True),
}


def as_float_bits(bits, width):
    """A float tensor whose raw bits are the given patterns (sign-bit patterns overflow a signed literal)."""
    signed = [b - (1 << width) if b >= 1 << (width - 1) else b for b in bits]
    if width == 32:
        return torch.tensor(signed, dtype=torch.int32).view(torch.float32)
    return torch.tensor(signed, dtype=torch.int16).view(torch.bfloat16)


def bit_view(tensor):
    """The tensor's raw bits as a same-width integer tensor, so negative zero and NaN compare exactly."""
    return tensor.view(torch.int16 if tensor.dtype == torch.bfloat16 else torch.int32)


def dest_bf16_bits(ieee_bits):
    """A BF16 datum as DEST stores it: sign, then the 7 mantissa bits, then the 8 exponent bits.

    The debug bus returns DEST's own bit layout. With a 32-bit DEST the read returns the high half
    of each datum, which carries the same three fields in the same order, so the words are the
    same in both DEST widths.
    """
    return (
        (ieee_bits & _BF16_SIGN_MASK)
        | ((ieee_bits & _BF16_EXP_MASK) >> _BF16_EXP_SHIFT)
        | ((ieee_bits & _BF16_MANT_MASK) << _BF16_MANT_BITS)
    )


def dest_row_words(bf16_row):
    """The 8 debug-bus words of a 16-datum BF16 row: datum 2i in the low half of word i."""
    bits = [dest_bf16_bits(int(b) & 0xFFFF) for b in bit_view(bf16_row).tolist()]
    return tuple(bits[2 * i] | (bits[2 * i + 1] << 16) for i in range(WORDS_PER_ROW))


def face_row(tile, face, row):
    """Row ``row`` of face ``face`` of a 32x32 tile laid out face by face, as the kernel reads it."""
    start = face * MAX_FACE_ELEMENTS + row * FACE_C_DIM
    return tile[start : start + FACE_C_DIM]


def tile_pair(formats):
    """One tile of random stimuli per operand: A is unpacked into SrcA, B is planted into DEST."""
    src_A, tile_cnt, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=INPUT_DIMENSIONS,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=INPUT_DIMENSIONS,
    )
    return src_A, src_B, tile_cnt


def make_config(formats, dest_acc, src_A, src_B, tile_cnt, dump):
    return TestConfig(
        "sources/dbg_get_array_row_srca_test.cpp",
        formats,
        templates=[dump],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt,
            tile_count_res=2,
        ),
        dest_acc=dest_acc,
    )


def dest_tile_from(res_from_L1, output_format):
    """The DEST tile the kernel read back, as a tensor of the output format."""
    assert (
        len(res_from_L1) == 2 * MAX_TILE_ELEMENTS
    ), f"Result length {len(res_from_L1)} does not match the two-tile layout the kernel writes"
    tile = res_from_L1[
        RESULT_TILE_DEST
        * MAX_TILE_ELEMENTS : (RESULT_TILE_DEST + 1)
        * MAX_TILE_ELEMENTS
    ]
    return torch.tensor(tile, dtype=format_dict[output_format])


def assert_dest_tile_restored(expected, res_tensor):
    # Only the borrowed row is at risk, so report on it directly rather than letting a whole-tile
    # mismatch hide which half came back wrong.
    differing = (
        (bit_view(expected) != bit_view(res_tensor)).nonzero().flatten().tolist()
    )
    assert not differing, (
        "dbg_get_array_row(SRCA) did not restore the dest row it borrowed. "
        f"{len(differing)} datum(s) differ, first at index {differing[0]}: "
        f"expected {expected[differing[0]]}, got {res_tensor[differing[0]]}. "
        "The save must hold both halves of the row, without converting them."
    )


@parametrize(
    formats=input_output_formats(
        [DataFormat.Float32, DataFormat.Float16_b, DataFormat.Int32],
        same=True,
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_srca_row_dump_restores_dest_row_0(formats, dest_acc):

    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("The RISC-DEST debug window is only available on Blackhole.")

    src_A, src_B, tile_cnt = tile_pair(formats)
    # The kernel plants buffer_B into DEST; put the at-risk patterns there.
    if formats.input_format == DataFormat.Int32:
        src_B[: len(INT32_AT_RISK)] = torch.tensor(INT32_AT_RISK, dtype=src_B.dtype)
    elif formats.input_format == DataFormat.Float32:
        src_B[: len(FP32_AT_RISK_BITS)] = as_float_bits(FP32_AT_RISK_BITS, 32)
    elif formats.input_format == DataFormat.Float16_b:
        src_B[: len(BF16_AT_RISK_BITS)] = as_float_bits(BF16_AT_RISK_BITS, 16)

    configuration = make_config(
        formats, dest_acc, src_A, src_B, tile_cnt, DBG_SRCA_DUMP(srca_row_check=False)
    )

    res_from_L1 = configuration.run().result
    res_tensor = dest_tile_from(res_from_L1, formats.output_format)
    expected = src_B.to(format_dict[formats.output_format])
    assert_dest_tile_restored(expected, res_tensor)


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_offset=list(DEST_OFFSETS),
)
def test_srca_row_dump_reads_srca_row(formats, dest_acc, dest_offset):

    if get_chip_architecture() == ChipArchitecture.QUASAR:
        pytest.skip("Quasar has no dbg_get_array_row.")

    dst_index, upper_half = DEST_OFFSETS[dest_offset]
    src_A, src_B, tile_cnt = tile_pair(formats)

    configuration = make_config(
        formats,
        dest_acc,
        src_A,
        src_B,
        tile_cnt,
        DBG_SRCA_DUMP(
            srca_row_check=True, dump_dst_index=dst_index, dump_upper_half=upper_half
        ),
    )

    res_from_L1 = configuration.run().result

    raw = configuration.variant_stimuli.collect_raw_result_bytes(
        TestConfig.TENSIX_LOCATION
    )
    srca_row = struct.unpack_from(f"<{WORDS_PER_ROW}I", raw, 0)
    offset_before, offset_after, offset_set = struct.unpack_from(
        "<3I", raw, 4 * DEST_OFFSET_WORD
    )

    # The datacopy consumed the tile face by face, alternating SrcA banks, and the dump reads the
    # bank used last: the last face. The expectation comes from the host tile rather than from a
    # DEST read-back, so it does not depend on how DEST happens to be addressed on the card.
    expected = dest_row_words(face_row(src_A, MAX_NUM_FACES - 1, 0))
    face_hits = [
        face
        for face in range(MAX_NUM_FACES)
        if dest_row_words(face_row(src_A, face, 0)) == srca_row
    ]
    if face_hits:
        got = f"it is row 0 of face {face_hits[0]} instead"
    elif srca_row == dest_row_words(face_row(src_B, 0, 0)):
        got = "it is row 0 of the tile planted in dest: a stale physical row was read back"
    elif not any(srca_row):
        got = "it is all zero: a stale, never-written physical row was read back"
    else:
        got = "it matches no row the test wrote"
    assert srca_row == expected, (
        f"dbg_get_array_row(SRCA) with the math dest offset at {dest_offset} returned "
        f"{[hex(w) for w in srca_row]}, expected SrcA row 0 of the last face "
        f"{[hex(w) for w in expected]} ({got})"
    )

    # The offset the dump found is the one the datacopy set (this also checks the read-back the
    # helper saves with), and the dump must hand it back unchanged.
    assert offset_before == offset_set, (
        f"the math thread's dest offset read before the dump is {offset_before}, "
        f"but the datacopy set it to {offset_set}"
    )
    assert offset_after == offset_before, (
        f"dbg_get_array_row(SRCA) left the math thread's dest offset at {offset_after}; "
        f"it was {offset_before} before the dump and must be handed back unchanged"
    )

    if get_chip_architecture() == ChipArchitecture.BLACKHOLE:
        # The row the helper borrowed, in the first tile of the current half, must have come back intact.
        res_tensor = dest_tile_from(res_from_L1, formats.output_format)
        expected_tile = src_B.to(format_dict[formats.output_format])
        assert_dest_tile_restored(expected_tile, res_tensor)
