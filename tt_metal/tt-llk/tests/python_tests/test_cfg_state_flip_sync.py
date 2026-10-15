# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
A configuration-bank flip followed by two NOPs needs no tensix_sync() before the thread's next
configuration instruction.

Each TRISC flips the hardware state ID, pads with 0..2 NOPs, optionally drains, and then issues a
read-modify-write (to bank 1) and an RDCFG (from bank 0) of a scratch word seeded differently in the
two banks. A consumer that sampled the state ID too early lands in, or reads from, the wrong bank.
A MOP of read-modify-writes to another byte of the same word can be queued ahead of each flip as a
backlog; those must land in the bank selected when they were issued. See
sources/cfg_state_flip_sync_test.cpp for the result layout.

Blackhole boots with instruction gathering on and both tt-metal and this harness turn it off; the
gathering-on case is run separately and only reported, because the ISA restricts what may follow a
state-ID write inside a fused bundle.
"""

from dataclasses import dataclass

import pytest
from conftest import blackhole_only
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TILE_COUNT, TemplateParameter
from loguru import logger

pytestmark = [blackhole_only]

RESULT_MAGIC = 0x43464753
WORDS_PER_THREAD = 256
THREADS = ["unpack", "math", "pack"]


@dataclass
class CFG_FLIP_GATHERING(TemplateParameter):
    cfg_flip_gathering_on: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool GATHERING_ON = {'true' if self.cfg_flip_gathering_on else 'false'};"


def _combination_label(combination):
    backlog, rest = divmod(combination, 6)
    nops, sync = divmod(rest, 2)
    return f"backlog={backlog} nops={nops} {'tensix_sync' if sync else 'no sync'}"


@pytest.mark.parametrize(
    "gathering_on", [False, True], ids=["gathering_off", "gathering_on"]
)
def test_cfg_state_flip_needs_no_drain(gathering_on):
    formats = input_output_formats([DataFormat.Int32])[0]
    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    configuration = TestConfig(
        "sources/cfg_state_flip_sync_test.cpp",
        formats,
        templates=[CFG_FLIP_GATHERING(gathering_on)],
        runtimes=[TILE_COUNT(tile_cnt_A)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1,
        ),
    )

    words = [int(w) & 0xFFFFFFFF for w in configuration.run().result]
    assert len(words) >= len(THREADS) * WORDS_PER_THREAD

    failures = []
    for thread_index, thread in enumerate(THREADS):
        base = thread_index * WORDS_PER_THREAD
        magic = words[base]
        assert magic == RESULT_MAGIC, f"{thread}: no result written (magic {magic:#x})"
        trials = words[base + 1]
        assert words[base + 2] == (1 if gathering_on else 0)

        for combination in range(12):
            rmw_wrong, rdcfg_wrong = words[
                base + 8 + 4 * combination : base + 10 + 4 * combination
            ]
            raw = words[base + 64 + 4 * combination : base + 67 + 4 * combination]
            label = _combination_label(combination)
            logger.info(
                f"{thread}: {label}: wrong bank on {rmw_wrong}/{trials} read-modify-writes and "
                f"{rdcfg_wrong}/{trials} RDCFGs (trial 0: bank0={raw[0]:#010x} bank1={raw[1]:#010x} rdcfg={raw[2]:#010x})"
            )
            if rmw_wrong or rdcfg_wrong:
                failures.append(f"{thread}: {label}: {rmw_wrong}/{rdcfg_wrong}")

    if gathering_on:
        # Reported only: shipped code never runs with gathering on.
        logger.info(f"gathering on: {len(failures)} combinations hit the wrong bank")
        return

    # With gathering off, every configuration instruction behind the flip uses the new bank whether or
    # not a drain sits between them, with any NOP padding, and with a backlog ahead of the flip.
    assert (
        not failures
    ), "wrong bank seen (thread: combination: rmw/rdcfg):\n" + "\n".join(failures)
