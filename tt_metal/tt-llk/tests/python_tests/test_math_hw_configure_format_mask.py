# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
The math thread's ALU source-format writers must mask the format value to the 4-bit register
field before shifting it in.

ALU_FORMAT_SPEC_REG0_SrcA and ALU_FORMAT_SPEC_REG1_SrcB are adjacent 4-bit fields of one config
word. The host passes UInt8 as 30 and UInt32 as 24, which have bit 4 set; a writer that shifts the
raw value lands that bit on the LSB of SrcB, and the two-field writers' RMW mask carries it into
the register, so SrcB is programmed as (srcB | 1): Int8 becomes Bfp2_b, Int32 becomes UInt16.

MATH programs each (srcA, srcB) pair through each writer and reads the config word back. See
sources/math_hw_configure_format_mask_test.cpp for the result layout.
"""

from collections import namedtuple

from conftest import wormhole_only
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig

pytestmark = [wormhole_only]

# Must match sources/math_hw_configure_format_mask_test.cpp
RESULT_MAGIC = 0x414C5546
HEADER_WORDS = 3
WORDS_PER_RECORD = 4
NUM_PAIRS = 7
WRITER_RECONFIG_SKIP_INT8 = 2
WRITERS = {
    0: "_llk_math_hw_configure_",
    1: "_llk_math_reconfig_data_format_",
    WRITER_RECONFIG_SKIP_INT8: "_llk_math_reconfig_data_format_<skip_int8>",
}

# tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines/cfg_defines.h
SRCA_SHAMT, SRCA_MASK = 17, 0x1E0000
SRCB_SHAMT, SRCB_MASK = 21, 0x1E00000
INT8_MATH_SHAMT, INT8_MATH_MASK = 31, 0x80000000

FIELD_WIDTH_MASK = 0xF  # DATA_FORMAT_BIT_COUNT = 4
INT8 = 14
INT32 = 8

# One result record: the writer index, the (srcA, srcB) formats it was given, and the
# ALU_FORMAT_SPEC_REG config word read back after it ran.
Record = namedtuple("Record", ["writer", "srca", "srcb", "word"])


def _int8_math_enabled(raw_format):
    # Mirrors is_int8_or_int32_format (ckernel_defs.h): Int8 is matched after masking (so UInt8 = 30
    # counts), Int32 only raw (so UInt32 = 24 does not enable INT8 math).
    return (raw_format & FIELD_WIDTH_MASK) == INT8 or raw_format == INT32


def test_math_alu_format_writers_mask_to_field_width():
    formats = input_output_formats([DataFormat.Int32])[0]
    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Int8/UInt8/Int32 math formats require FP32 dest accumulation.
    configuration = TestConfig(
        "sources/math_hw_configure_format_mask_test.cpp",
        formats,
        dest_acc=DestAccumulation.Yes,
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1,
            twos_complement=True,
        ),
    )

    words = [int(w) & 0xFFFFFFFF for w in configuration.run().result]

    magic, num_records, words_per_record = words[:HEADER_WORDS]
    assert magic == RESULT_MAGIC, f"no result written (magic {magic:#x})"
    assert words_per_record == WORDS_PER_RECORD
    assert len(words) >= HEADER_WORDS + num_records * words_per_record
    assert (
        num_records == len(WRITERS) * NUM_PAIRS
    ), f"unexpected record count {num_records}"

    records = [
        Record(*words[base : base + words_per_record])
        for base in range(
            HEADER_WORDS,
            HEADER_WORDS + num_records * words_per_record,
            words_per_record,
        )
    ]

    # The pair table must keep at least one SrcA whose value does not fit the field, or the
    # spill this test exists for is never exercised.
    srca_values = [record.srca for record in records]
    assert any(
        srca > FIELD_WIDTH_MASK for srca in srca_values
    ), f"no SrcA format wider than the field in the pair table: {srca_values}"

    failures = []
    for writer, srca, srcb, word in records:
        got_srca = (word & SRCA_MASK) >> SRCA_SHAMT
        got_srcb = (word & SRCB_MASK) >> SRCB_SHAMT
        got_int8 = (word & INT8_MATH_MASK) >> INT8_MATH_SHAMT

        want_srca = srca & FIELD_WIDTH_MASK
        want_srcb = srcb & FIELD_WIDTH_MASK
        # The skip_int8 writer leaves the INT8 bit as seeded (Float32 x Float32 -> 0).
        want_int8 = (
            0
            if writer == WRITER_RECONFIG_SKIP_INT8
            else int(_int8_math_enabled(srca) or _int8_math_enabled(srcb))
        )

        if (got_srca, got_srcb, got_int8) != (want_srca, want_srcb, want_int8):
            failures.append(
                f"{WRITERS[writer]}(srcA={srca}, srcB={srcb}): word {word:#010x} -> "
                f"SrcA={got_srca} SrcB={got_srcb} INT8={got_int8}, "
                f"expected SrcA={want_srca} SrcB={want_srcb} INT8={want_int8}"
            )

    assert not failures, "ALU format field(s) programmed wrong:\n" + "\n".join(failures)
