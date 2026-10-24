# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Regression test for ``xmov_wait_till_idle``: once it returns, the move issued just before it has
been carried out, so the destination can be read straight away.

Each iteration PACK writes a fresh pattern to the source buffer, issues an L1-to-L1 move through
``xmov_l1_to_l1_non_compact``, calls ``xmov_wait_till_idle`` and reads the destination back. An
iteration in which any destination word does not yet hold the pattern of this iteration is a
stale read. The last destination line is loaded right before the move, so on Blackhole it sits in
the RISC data cache with the previous iteration's contents whatever the move size, and the read
back only passes if the wait dropped the cache.

``xfer_16b`` is the move size in 16-byte units; ``gap`` is the number of ``nop``s between the wait
returning and the first destination read.
"""

import pytest
from conftest import skip_for_quasar
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import ITERATIONS, XMOV_READ_GAP, XMOV_XFER_16B

# Passed to the kernel as ITERATIONS; the kernel echoes it back as its first result word.
NUM_ITERATIONS = 8192
# cfg0 bit 3 (DisLowCash): set means the Blackhole L0 data cache is disabled.
CFG0_L0_DCACHE_DISABLE = 1 << 3


@skip_for_quasar
@pytest.mark.parametrize("gap", [0, 64])
@pytest.mark.parametrize("xfer_16b", [1, 16, 256])
def test_xmov_wait_till_idle(xfer_16b, gap):
    if TestConfig.TEST_TARGET.run_simulator:
        pytest.skip(
            "The TDMA-RISC mover and its status bits are unverified on the simulator; "
            "the wait would spin to the timeout if they are not modelled."
        )
    formats = input_output_formats([DataFormat.Int32])[0]
    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    configuration = TestConfig(
        "sources/xmov_wait_till_idle_test.cpp",
        formats,
        templates=[
            ITERATIONS(NUM_ITERATIONS),
            XMOV_XFER_16B(xfer_16b),
            XMOV_READ_GAP(gap),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            twos_complement=True,
        ),
    )
    # Raw u32 words.
    vals = [int(v) & 0xFFFFFFFF for v in configuration.run().result]
    (
        iterations,
        stale_iterations,
        first_stale_iter,
        first_stale_word,
        first_stale_words,
        src_mismatches,
        kernel_xfer_16b,
        kernel_gap,
        cfg0,
    ) = vals[:9]

    print(
        f"\nxmov_wait_till_idle, {xfer_16b} x 16 B per move, {gap} nops before the read"
        + (
            f", cfg0 {cfg0:#x} (L0 data cache {'disabled' if cfg0 & CFG0_L0_DCACHE_DISABLE else 'enabled'})"
            if get_chip_architecture() == ChipArchitecture.BLACKHOLE
            else ""
        )
        + ":"
        f"\n  stale destination reads: {stale_iterations}/{iterations} iterations"
        + (
            f" (first at iteration {first_stale_iter}, word {first_stale_word}, "
            f"{first_stale_words}/{xfer_16b * 4} words stale)"
            if stale_iterations
            else ""
        )
    )

    assert (
        iterations == NUM_ITERATIONS
    ), f"kernel ran {iterations} iterations, expected {NUM_ITERATIONS}"
    assert (kernel_xfer_16b, kernel_gap) == (
        xfer_16b,
        gap,
    ), "kernel was built with other parameters"
    assert (
        src_mismatches == 0
    ), f"the source pattern was not in L1 before the move: {src_mismatches} mismatching words"
    assert stale_iterations == 0, (
        f"xmov_wait_till_idle returned before the move was done in {stale_iterations}/{iterations} "
        f"iterations (first at iteration {first_stale_iter}: word {first_stale_word} and "
        f"{first_stale_words - 1} more did not yet hold the pattern of that iteration)"
    )
    if (
        get_chip_architecture() == ChipArchitecture.BLACKHOLE
        and cfg0 & CFG0_L0_DCACHE_DISABLE
    ):
        pytest.skip(
            f"the L0 data cache was disabled at boot (cfg0 {cfg0:#x}), so the cache invalidate "
            "in xmov_wait_till_idle was not exercised"
        )
