# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
tensix_sync() on its own must wait for a MOP that is still expanding.

Each TRISC runs a MOP whose only instruction increments a GPR, synchronises one of four ways
(no sync, tensix_sync, mop_sync, mop_sync then tensix_sync) and reads the GPR back from the RISC.
A sync that returned while the MOP was still expanding shows up as a read short of the programmed
number of increments. See sources/tensix_sync_mop_drain_test.cpp for the result layout.
"""

from conftest import blackhole_only
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TILE_COUNT
from loguru import logger

pytestmark = [blackhole_only]

RESULT_MAGIC = 0x4D4F5053
WORDS_PER_THREAD = 256
THREADS = ["unpack", "math", "pack"]
ARMS = ["no sync", "tensix_sync", "mop_sync", "mop_sync + tensix_sync"]


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

    for thread_index, thread in enumerate(THREADS):
        base = thread_index * WORDS_PER_THREAD
        magic = words[base]
        assert magic == RESULT_MAGIC, f"{thread}: no result written (magic {magic:#x})"

        increments = words[base + 1]
        long_counts = words[base + 2 : base + 6]
        long_cycles = words[base + 6 : base + 10]
        short_misses = words[base + 10 : base + 14]
        trials = words[base + 14]
        for arm, count, cycles, misses in zip(
            ARMS, long_counts, long_cycles, short_misses
        ):
            logger.info(
                f"{thread}: {arm}: long MOP read {count}/{increments} after {cycles} cycles; "
                f"short MOP {misses}/{trials} early reads"
            )

        no_sync, tensix_only, mop_only, both = long_counts
        # The probe can see an unfinished MOP: without any sync the read lands long before the
        # long MOP has finished.
        assert (
            no_sync < increments
        ), f"{thread}: no-sync read already saw the whole long MOP"
        # tensix_sync alone waits for the whole MOP, exactly as the mop_sync + tensix_sync pair does.
        assert (
            tensix_only == increments
        ), f"{thread}: tensix_sync returned with the long MOP unfinished"
        assert (
            both == increments
        ), f"{thread}: mop_sync + tensix_sync returned with the long MOP unfinished"
        assert (
            short_misses[1] == 0
        ), f"{thread}: tensix_sync returned early on {short_misses[1]} short MOP runs"
        assert (
            short_misses[3] == 0
        ), f"{thread}: mop_sync + tensix_sync returned early on {short_misses[3]} runs"
        # mop_sync alone is only reported: it waits for the MOP decoder, not for the generated
        # instructions to retire, so it may legitimately read short.
        assert mop_only <= increments
