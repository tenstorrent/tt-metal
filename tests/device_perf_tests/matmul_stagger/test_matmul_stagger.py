# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import pytest
from loguru import logger

from models.common.utility_functions import run_for_wormhole_b0_or_blackhole
from tracy.process_model_log import post_process_ops_log, run_device_profiler
from tests.device_perf_tests.matmul_stagger.test_run_8x7_matmul import (
    MATMUL_VARIANTS,
    MULTI_BLOCK_INNER_BLOCKS,
    MULTI_BLOCK_MATMUL_VARIANT,
)

# The stagger delays the odd-row cores by TT_MM_STAGGER_VALUE cycles after each compute-kernel wait on in1
# (tt_metal/hw/ckernels/<arch>/metal/llk_io/llk_io_unpack.h); the matmul kernels profiled here wait once per in1
# block. Type 1 delays the first block only, type 2 every block, type 3 the first block and every
# STAGGER_PERIOD_IN_BLOCKS-th after it.
STAGGER_TYPE_ENV = "TT_MM_STAGGER_TYPE"
STAGGER_VALUE_ENV = "TT_MM_STAGGER_VALUE"
STAGGER_ENV_VARS = (STAGGER_TYPE_ENV, STAGGER_VALUE_ENV)
# TT_MM_STAGGER_TYPE codes, as the host passes them to the kernel.
FIRST_BLOCK_ONLY = "1"
EVERY_BLOCK = "2"
PERIODIC = "3"
STAGGER_TYPES = [FIRST_BLOCK_ONLY, EVERY_BLOCK, PERIODIC]
# Mirrors stagger_period_in_blocks in llk_io_unpack.h.
STAGGER_PERIOD_IN_BLOCKS = 20
DURATION_COLUMN = "DEVICE KERNEL DURATION [ns]"

# Single-block cases: one million cycles (at least 500 us on any Tensix clock) makes the op slower by far more
# than its run-to-run noise whichever type is selected.
STAGGER_CYCLES = 1_000_000
MIN_STAGGER_DELTA_NS = 250_000

# Multi-block case: a shorter delay keeps the type-2 total (one delay per inner block) at a few milliseconds.
MULTI_BLOCK_STAGGER_CYCLES = 250_000
# Blocks each type delays on an odd-row core of the multi-block variant: type 2 all of them, type 3 blocks
# 0, 20, 40, ..., type 1 block 0 only.
EXPECTED_STAGGERED_BLOCKS = {
    FIRST_BLOCK_ONLY: 1,
    EVERY_BLOCK: MULTI_BLOCK_INNER_BLOCKS,
    PERIODIC: -(-MULTI_BLOCK_INNER_BLOCKS // STAGGER_PERIOD_IN_BLOCKS),
}
# Relative tolerance on the per-type delta against (staggered blocks x per-block delay). Tight enough that a
# wrong type-3 period (2 or 5 delays instead of 3) or a type 1 that fires on every block fails.
BLOCK_COUNT_TOLERANCE = 0.3


def profile_kernel_duration_ns(matmul_variant, output_logs_subdir, stagger_type=None, stagger_cycles=None):
    """Profile one matmul variant in a subprocess and return its device kernel duration in ns.

    The profiled run inherits this process's environment: the stagger variables are set for a with-stagger
    run and cleared for a baseline run, and the caller's own settings are handed back whatever happens.
    """
    command = f"pytest tests/device_perf_tests/matmul_stagger/test_run_8x7_matmul.py -k {matmul_variant}"
    saved_stagger_env = {key: os.environ.get(key) for key in STAGGER_ENV_VARS}
    try:
        if stagger_type is None:
            for key in STAGGER_ENV_VARS:
                os.environ.pop(key, None)
        else:
            os.environ[STAGGER_TYPE_ENV] = stagger_type
            os.environ[STAGGER_VALUE_ENV] = str(stagger_cycles)
        run_device_profiler(command, output_logs_subdir)
    finally:
        for key, value in saved_stagger_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return post_process_ops_log(output_logs_subdir, [DURATION_COLUMN])[DURATION_COLUMN]


# No-stagger duration per matmul variant, profiled once per session: it does not depend on the stagger type, so
# every type is compared against the same baseline.
_baseline_duration_ns = {}


def baseline_kernel_duration_ns(matmul_variant):
    if matmul_variant not in _baseline_duration_ns:
        _baseline_duration_ns[matmul_variant] = profile_kernel_duration_ns(
            matmul_variant, f"test_matmul_stagger_{matmul_variant}"
        )
    return _baseline_duration_ns[matmul_variant]


# Hook-is-consumed smoke test: every variant here has one in1 block per core, so types 1, 2 and 3 all delay
# that one block and these cases only show that each MM_STAGGER_TYPE arm is compiled in and slows the op.
# The per-type firing pattern is checked by test_matmul_stagger_block_count below.
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("stagger_type", STAGGER_TYPES)
@pytest.mark.parametrize(
    "matmul_variant",
    MATMUL_VARIANTS,
)
def test_matmul_stagger(matmul_variant, stagger_type):
    output_logs_subdir_with_stagger = f"test_matmul_stagger_{matmul_variant}_with_stagger_type{stagger_type}"

    duration_with = profile_kernel_duration_ns(
        matmul_variant, output_logs_subdir_with_stagger, stagger_type, STAGGER_CYCLES
    )
    duration_without = baseline_kernel_duration_ns(matmul_variant)
    logger.info(
        f"{matmul_variant} stagger type {stagger_type}: {DURATION_COLUMN} with stagger = {duration_with:.0f}, "
        f"without = {duration_without:.0f}"
    )
    assert duration_with - duration_without > MIN_STAGGER_DELTA_NS, (
        f"Stagger type {stagger_type} with {STAGGER_CYCLES} cycles should make {matmul_variant} visibly slower: "
        f"with stagger {duration_with:.0f} ns, without {duration_without:.0f} ns"
    )


# Block-count test: on the 41-inner-block variant the slowdown over the no-stagger run scales with the number
# of delayed blocks, so the three types separate (type 2 > type 3 > type 1) and each delta must match its
# block count times the per-block delay taken from type 2.
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.models_device_performance_bare_metal
def test_matmul_stagger_block_count():
    duration_without = baseline_kernel_duration_ns(MULTI_BLOCK_MATMUL_VARIANT)
    deltas = {}
    for stagger_type in STAGGER_TYPES:
        duration_with = profile_kernel_duration_ns(
            MULTI_BLOCK_MATMUL_VARIANT,
            f"test_matmul_stagger_{MULTI_BLOCK_MATMUL_VARIANT}_with_stagger_type{stagger_type}",
            stagger_type,
            MULTI_BLOCK_STAGGER_CYCLES,
        )
        deltas[stagger_type] = duration_with - duration_without
        logger.info(
            f"{MULTI_BLOCK_MATMUL_VARIANT} stagger type {stagger_type}: {DURATION_COLUMN} with stagger = "
            f"{duration_with:.0f}, without = {duration_without:.0f}, delta = {deltas[stagger_type]:.0f}"
        )

    assert deltas[EVERY_BLOCK] > deltas[PERIODIC] > deltas[FIRST_BLOCK_ONLY] > 0, (
        f"Stagger deltas should order type {EVERY_BLOCK} (every block) > type {PERIODIC} "
        f"(every {STAGGER_PERIOD_IN_BLOCKS}th block) > type {FIRST_BLOCK_ONLY} (first block) > 0 on "
        f"{MULTI_BLOCK_INNER_BLOCKS} inner blocks: {deltas}"
    )
    delay_per_block_ns = deltas[EVERY_BLOCK] / EXPECTED_STAGGERED_BLOCKS[EVERY_BLOCK]
    for stagger_type in (FIRST_BLOCK_ONLY, PERIODIC):
        expected_delta = EXPECTED_STAGGERED_BLOCKS[stagger_type] * delay_per_block_ns
        assert abs(deltas[stagger_type] - expected_delta) <= BLOCK_COUNT_TOLERANCE * expected_delta, (
            f"Stagger type {stagger_type} should delay {EXPECTED_STAGGERED_BLOCKS[stagger_type]} of "
            f"{MULTI_BLOCK_INNER_BLOCKS} blocks ({expected_delta:.0f} ns at {delay_per_block_ns:.0f} ns per block), "
            f"measured delta {deltas[stagger_type]:.0f} ns"
        )
