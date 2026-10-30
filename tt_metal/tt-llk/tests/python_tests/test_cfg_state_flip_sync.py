# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
A configuration-bank flip followed by two NOPs needs no tensix_sync() before the thread's next
configuration instruction.

Each TRISC flips the hardware state ID, pads with 0..2 NOPs, optionally drains, and then issues a
read-modify-write (to bank 1) and an RDCFG (from bank 0) of a word seeded differently in the two
banks, with or without a MOP of read-modify-writes queued ahead of the flip. A consumer that
sampled the state ID too early lands in, or reads from, the wrong bank. See
sources/cfg_state_flip_sync_test.cpp for the result layout.
"""

from conftest import blackhole_only
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from loguru import logger

pytestmark = [blackhole_only]

# Must match sources/cfg_state_flip_sync_test.cpp
RESULT_MAGIC = 0x43464753
WORDS_PER_THREAD = 256
TRIALS_WORD = 1
COUNT_WORDS = 8  # 4 words per combination: rmw wrong, rdcfg wrong, reserved, reserved
RAW_WORDS = 64  # 4 words per combination: trial-0 bank 0, bank 1, RDCFG value, reserved
WORDS_PER_COMBO = 4
COMBINATIONS = 12  # backlog {0,1} x nops {0,1,2} x sync {0,1}

THREADS = ["unpack", "math", "pack"]


def _combination_label(combination):
    backlog, rest = divmod(combination, 6)
    nops, sync = divmod(rest, 2)
    return f"backlog={backlog} nops={nops} {'tensix_sync' if sync else 'no sync'}"


def test_cfg_state_flip_needs_no_drain():
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
    assert len(words) >= len(THREADS) * WORDS_PER_THREAD

    failures = []
    for thread_index, thread in enumerate(THREADS):
        base = thread_index * WORDS_PER_THREAD
        region = words[base : base + WORDS_PER_THREAD]
        magic = region[0]
        assert magic == RESULT_MAGIC, f"{thread}: no result written (magic {magic:#x})"
        trials = region[TRIALS_WORD]

        for combination in range(COMBINATIONS):
            counts = COUNT_WORDS + WORDS_PER_COMBO * combination
            rmw_wrong, rdcfg_wrong = region[counts : counts + 2]
            raw = RAW_WORDS + WORDS_PER_COMBO * combination
            bank0, bank1, rdcfg = region[raw : raw + 3]
            label = _combination_label(combination)
            logger.info(
                f"{thread}: {label}: wrong bank on {rmw_wrong}/{trials} read-modify-writes and "
                f"{rdcfg_wrong}/{trials} RDCFGs (trial 0: bank0={bank0:#010x} bank1={bank1:#010x} rdcfg={rdcfg:#010x})"
            )
            if rmw_wrong or rdcfg_wrong:
                failures.append(f"{thread}: {label}: {rmw_wrong}/{rdcfg_wrong}")

    assert (
        not failures
    ), "wrong bank seen (thread: combination: rmw/rdcfg):\n" + "\n".join(failures)
