# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Regression test for ``mailbox_read``: it returns only once the message has arrived, even when
the caller never uses the value.

Each cycle MATH programs a config field and then releases UNPACK and PACK through their mailboxes.
The released threads discard the message and read the field back, which must already hold the
value MATH programmed in that cycle.
"""

import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig

# Must match sources/mailbox_read_completion_test.cpp
CYCLES = 256


@pytest.mark.skipif(
    get_chip_architecture()
    not in (ChipArchitecture.BLACKHOLE, ChipArchitecture.WORMHOLE),
    reason="The test kernel uses the Blackhole and Wormhole config layout.",
)
def test_mailbox_read_completion():
    formats = input_output_formats([DataFormat.Int32])[0]
    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    configuration = TestConfig(
        "sources/mailbox_read_completion_test.cpp",
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
    vals = [int(v) for v in configuration.run().result]
    cycles, pack_seen, unpack_seen = vals[:3]

    assert cycles == CYCLES, f"kernel reported {cycles} cycles, expected {CYCLES}"
    stale = {
        name: count
        for name, count in {"PACK": pack_seen, "UNPACK": unpack_seen}.items()
        if count != CYCLES
    }
    assert not stale, "mailbox_read returned before the message arrived: " + ", ".join(
        f"{name} saw MATH's value in {count}/{CYCLES} cycles"
        for name, count in stale.items()
    )
