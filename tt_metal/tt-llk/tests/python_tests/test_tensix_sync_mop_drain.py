# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
tensix_sync() on its own must wait for a MOP that is still expanding.

Each TRISC runs a MOP whose only instruction increments a GPR, with or without a tensix_sync()
behind it, and reads the GPR back from the RISC. The no-sync arm shows the probe can see an
unfinished MOP; it is asserted only on the long MOP, where a million increments leave no margin
for timing. See sources/tensix_sync_mop_drain_test.cpp for the result layout.
"""

from conftest import blackhole_only
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from loguru import logger

pytestmark = [blackhole_only]

# Must match sources/tensix_sync_mop_drain_test.cpp
RESULT_MAGIC = 0x4D4F5053
WORDS_PER_THREAD = 256
LONG_INCREMENTS_WORD = 1
LONG_COUNT_WORDS = slice(2, 4)
SHORT_MISS_WORDS = slice(4, 6)
SHORT_TRIALS_WORD = 6
SHORT_INCREMENTS_WORD = 7

THREADS = ["unpack", "math", "pack"]
ARMS = ["no sync", "tensix_sync"]
NO_SYNC, TENSIX_SYNC = range(2)


def test_tensix_sync_drains_in_flight_mop():
    formats = input_output_formats([DataFormat.Int32])[0]
    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    configuration = TestConfig(
        "sources/tensix_sync_mop_drain_test.cpp",
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
        # The no-sync reads are deliberately unsynchronised.
        expected_nondeterministic=True,
    )

    words = [int(w) & 0xFFFFFFFF for w in configuration.run().result]
    assert len(words) >= len(THREADS) * WORDS_PER_THREAD

    for thread_index, thread in enumerate(THREADS):
        base = thread_index * WORDS_PER_THREAD
        region = words[base : base + WORDS_PER_THREAD]
        magic = region[0]
        assert magic == RESULT_MAGIC, f"{thread}: no result written (magic {magic:#x})"

        increments = region[LONG_INCREMENTS_WORD]
        long_counts = region[LONG_COUNT_WORDS]
        short_misses = region[SHORT_MISS_WORDS]
        trials = region[SHORT_TRIALS_WORD]
        short_increments = region[SHORT_INCREMENTS_WORD]
        for arm, count, misses in zip(ARMS, long_counts, short_misses):
            logger.info(
                f"{thread}: {arm}: long MOP read {count}/{increments}; "
                f"short MOP ({short_increments} increments) read short on {misses}/{trials} runs"
            )

        assert (
            long_counts[NO_SYNC] < increments
        ), f"{thread}: no-sync read already saw the whole long MOP"
        assert (
            long_counts[TENSIX_SYNC] == increments
        ), f"{thread}: tensix_sync returned with the long MOP unfinished"
        assert (
            short_misses[TENSIX_SYNC] == 0
        ), f"{thread}: tensix_sync returned early on {short_misses[TENSIX_SYNC]} short MOP runs"
