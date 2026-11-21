# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Wormhole: the math thread's ALU source-format writers must mask each DataFormat to its 4-bit field, or
bit 4 of SrcA (UInt8 = 30, UInt32 = 24) spills onto the LSB of SrcB (srcB | 1). Reads the config word
back after each writer; result layout in sources/math_hw_configure_format_mask_test.cpp.
"""

from collections import namedtuple

from conftest import wormhole_only
from helpers.format_config import WORMHOLE_DATA_FORMAT_ENUM_VALUES, DataFormat
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
NUM_PAIRS = 8
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

# The records carry the raw DataFormat values the kernel passed to the writers.
FORMAT_BY_RAW_VALUE = {
    raw: fmt for fmt, raw in WORMHOLE_DATA_FORMAT_ENUM_VALUES.items()
}

# One result record: the writer index, the (srcA, srcB) formats it was given, and the
# ALU_FORMAT_SPEC_REG config word read back after it ran.
Record = namedtuple("Record", ["writer", "srca", "srcb", "word"])


def _int8_math_enabled(raw_format):
    # Int8/UInt8/Int32 enable INT8 math, UInt32 does not (is_int8_or_int32_format in ckernel_defs.h).
    return FORMAT_BY_RAW_VALUE[raw_format].needs_int8_math_config()


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
        # The skip_int8 writer is seeded with Int8 x Float16_b (INT8 = 1) and must leave the bit as
        # it found it; the other writers are seeded with INT8 = 0 and must re-derive it.
        want_int8 = (
            1
            if writer == WRITER_RECONFIG_SKIP_INT8
            else int(_int8_math_enabled(srca) or _int8_math_enabled(srcb))
        )

        if (got_srca, got_srcb, got_int8) != (want_srca, want_srcb, want_int8):
            failures.append(
                f"{WRITERS[writer]}(srcA={FORMAT_BY_RAW_VALUE[srca]}={srca}, "
                f"srcB={FORMAT_BY_RAW_VALUE[srcb]}={srcb}): word {word:#010x} -> "
                f"SrcA={got_srca} SrcB={got_srcb} INT8={got_int8}, "
                f"expected SrcA={want_srca} SrcB={want_srcb} INT8={want_int8}"
            )

    assert not failures, "ALU format field(s) programmed wrong:\n" + "\n".join(failures)
