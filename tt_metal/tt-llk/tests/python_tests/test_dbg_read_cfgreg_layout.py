# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
``dbg_read_cfgreg(bank, addr)`` must return the configuration word it names.

The debug selector indexes the RISC's word view of the configuration space (state bank 0,
state bank 1, the dual-write window, then one bank per thread), so the base added for
"state bank 1" and "thread n" has to follow this architecture's bank sizes. The math thread
plants a known value in a bank-1 word, in a bank-0 word (control) and in its own thread-CFG
word, reads each back through ``dbg_read_cfgreg`` and restores it. See
sources/dbg_read_cfgreg_layout_test.cpp for the result layout.
"""

from conftest import blackhole_only
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from loguru import logger

pytestmark = [blackhole_only]

# Must match sources/dbg_read_cfgreg_layout_test.cpp
RESULT_MAGIC = 0x43464744
PROBE_COUNT_WORD = 1
PROBE_BASE = 8
WORDS_PER_PROBE = 4
PROBE_ID_BANK_SHIFT = 16
EXPECTED_PROBES = 4  # two HW_CFG_1 words, the HW_CFG_0 control, the thread word

BANK_NAMES = {
    0: "THREAD_0_CFG",
    1: "THREAD_1_CFG",
    2: "THREAD_2_CFG",
    3: "HW_CFG_0",
    4: "HW_CFG_1",
}


def test_dbg_read_cfgreg_reads_the_named_word():
    formats = input_output_formats([DataFormat.Int32])[0]
    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    configuration = TestConfig(
        "sources/dbg_read_cfgreg_layout_test.cpp",
        formats,
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
    assert words[0] == RESULT_MAGIC, f"no result written (magic {words[0]:#x})"
    probe_count = words[PROBE_COUNT_WORD]
    assert (
        probe_count == EXPECTED_PROBES
    ), f"the kernel recorded {probe_count} probes, expected {EXPECTED_PROBES}"

    mismatches = []
    for probe in range(probe_count):
        base = PROBE_BASE + WORDS_PER_PROBE * probe
        ident, expected, got = words[base : base + 3]
        bank = ident >> PROBE_ID_BANK_SHIFT
        addr = ident & ((1 << PROBE_ID_BANK_SHIFT) - 1)
        label = f"dbg_read_cfgreg({BANK_NAMES.get(bank, bank)}, ADDR32 {addr})"
        logger.info(f"{label}: planted {expected:#010x}, read {got:#010x}")
        if got != expected:
            mismatches.append(
                f"{label} returned {got:#010x}, the word holds {expected:#010x}"
            )

    assert (
        not mismatches
    ), "dbg_read_cfgreg landed on the wrong configuration word:\n" + "\n".join(
        mismatches
    )
