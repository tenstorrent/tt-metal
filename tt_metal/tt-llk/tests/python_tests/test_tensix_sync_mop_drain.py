# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
tensix_sync() on its own must wait for a MOP that is still expanding.

Each TRISC runs a MOP whose only instruction increments a GPR, synchronises one of four ways
(no sync, tensix_sync, mop_sync, mop_sync then tensix_sync) and reads the GPR back from the RISC.
A sync that returned while the MOP was still expanding shows up as a read short of the programmed
number of increments. See sources/tensix_sync_mop_drain_test.cpp for the result layout.

Only the two tensix_sync arms are asserted. The no-sync and mop_sync arms are reported: what they
read depends on how far the MOP got before the read landed, which is what makes the probe able
to see an unfinished MOP, and the long MOP's no-sync read is asserted short only because a
million increments leave no margin for timing.
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
LONG_COUNT_WORDS = slice(2, 6)
LONG_CYCLE_WORDS = slice(6, 10)
SHORT_MISS_WORDS = slice(10, 14)
SHORT_TRIALS_WORD = 14
SHORT_INCREMENTS_WORD = 15

THREADS = ["unpack", "math", "pack"]
ARMS = ["no sync", "tensix_sync", "mop_sync", "mop_sync + tensix_sync"]
NO_SYNC, TENSIX_SYNC, MOP_SYNC, BOTH = range(4)


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
        # The result holds wall-clock cycle counts and deliberately unsynchronised reads.
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
        long_cycles = region[LONG_CYCLE_WORDS]
        short_misses = region[SHORT_MISS_WORDS]
        trials = region[SHORT_TRIALS_WORD]
        short_increments = region[SHORT_INCREMENTS_WORD]
        for arm, count, cycles, misses in zip(
            ARMS, long_counts, long_cycles, short_misses
        ):
            logger.info(
                f"{thread}: {arm}: long MOP read {count}/{increments} after {cycles} cycles; "
                f"short MOP ({short_increments} increments) read short on {misses}/{trials} runs"
            )

        # The probe can see an unfinished MOP: without any sync the read lands long before the
        # long MOP has finished.
        assert (
            long_counts[NO_SYNC] < increments
        ), f"{thread}: no-sync read already saw the whole long MOP"
        # tensix_sync alone waits for the whole MOP, exactly as the mop_sync + tensix_sync pair does.
        assert (
            long_counts[TENSIX_SYNC] == increments
        ), f"{thread}: tensix_sync returned with the long MOP unfinished"
        assert (
            long_counts[BOTH] == increments
        ), f"{thread}: mop_sync + tensix_sync returned with the long MOP unfinished"
        assert (
            short_misses[TENSIX_SYNC] == 0
        ), f"{thread}: tensix_sync returned early on {short_misses[TENSIX_SYNC]} short MOP runs"
        assert (
            short_misses[BOTH] == 0
        ), f"{thread}: mop_sync + tensix_sync returned early on {short_misses[BOTH]} short MOP runs"
        # mop_sync alone waits for the MOP decoder, not for the generated instructions to retire,
        # so it may legitimately read short; it is reported, not asserted.
        assert long_counts[MOP_SYNC] <= increments
