# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``dbg_read_cfgreg(bank, addr)`` must return the configuration word the caller named, in
every bank.

The selector it writes to the debug read register is a word index into the memory-mapped
configuration window: two thread-agnostic banks of ``CFG_STATE_SIZE * 4`` words, then three
thread banks of ``THD_STATE_SIZE`` 16-byte entries. Only ``HW_CFG_0`` starts at word 0, so only
``HW_CFG_0`` reads are independent of the bank sizes the helper assumes.

The kernel plants a known value in every bank (a sentinel in the inactive thread-agnostic bank,
one ``SETC16`` per TRISC into its own thread bank) and reads each back through the helper. Thread
entries are compared on their 16-bit value, thread-agnostic words on the whole word.
"""

import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig

# Must match sources/dbg_read_cfgreg_test.cpp
NUM_CHECKS = 5
BANK_NAMES = {
    0: "THREAD_0_CFG",
    1: "THREAD_1_CFG",
    2: "THREAD_2_CFG",
    3: "HW_CFG_0",
    4: "HW_CFG_1",
}
THREAD_BANKS = (0, 1, 2)


@pytest.mark.skipif(
    get_chip_architecture() != ChipArchitecture.WORMHOLE,
    reason="The test kernel plants its sentinels through the Wormhole config layout.",
)
def test_dbg_read_cfgreg_every_bank():
    formats = input_output_formats([DataFormat.Int32])[0]
    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    configuration = TestConfig(
        "sources/dbg_read_cfgreg_test.cpp",
        formats,
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
    )
    vals = [int(v) & 0xFFFFFFFF for v in configuration.run().result]

    num_checks = vals[0]
    assert (
        num_checks == NUM_CHECKS
    ), f"kernel reported {num_checks} checks, expected {NUM_CHECKS}"

    mismatches = []
    for i in range(num_checks):
        bank, addr, expected, raw = vals[1 + 4 * i : 1 + 4 * i + 4]
        got = raw & 0xFFFF if bank in THREAD_BANKS else raw
        if got != expected:
            mismatches.append(
                f"{BANK_NAMES.get(bank, bank)} addr {addr}: expected 0x{expected:x}, "
                f"read 0x{got:x} (raw word 0x{raw:08x})"
            )

    assert not mismatches, (
        "dbg_read_cfgreg returned a different configuration word than the one named:\n  "
        + "\n  ".join(mismatches)
    )
