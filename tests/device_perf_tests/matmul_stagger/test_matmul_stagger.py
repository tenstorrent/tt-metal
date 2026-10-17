# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import pytest
from loguru import logger

from models.common.utility_functions import run_for_wormhole_b0_or_blackhole
from tracy.process_model_log import post_process_ops_log, run_device_profiler
from tests.device_perf_tests.matmul_stagger.test_run_8x7_matmul import MATMUL_VARIANTS

# The stagger delays the odd-row cores by TT_MM_STAGGER_VALUE cycles before an in1 block
# (tt_metal/hw/ckernels/<arch>/metal/llk_io/llk_io_unpack.h): type 1 before the first block only,
# type 2 before every block, type 3 before every 20th block. Every type delays the first block, so
# one million cycles (at least 500 us on any Tensix clock) makes the op slower by far more than its
# run-to-run noise whichever type is selected.
STAGGER_TYPES = ["1", "2", "3"]
STAGGER_CYCLES = 1_000_000
MIN_STAGGER_DELTA_NS = 250_000
STAGGER_ENV_VARS = ("TT_MM_STAGGER_TYPE", "TT_MM_STAGGER_VALUE")


# for all matmul variants that can enable stagger, profile op with stagger enabled and disabled,
# and make sure the execution time of the op is longer with stagger enabled
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("stagger_type", STAGGER_TYPES)
@pytest.mark.parametrize(
    "matmul_variant",
    MATMUL_VARIANTS,
)
def test_matmul_stagger(matmul_variant, stagger_type):
    duration_column = "DEVICE KERNEL DURATION [ns]"
    output_logs_subdir = f"test_matmul_stagger_{matmul_variant}"
    output_logs_subdir_with_stagger = f"{output_logs_subdir}_with_stagger_type{stagger_type}"

    command = f"pytest tests/device_perf_tests/matmul_stagger/test_run_8x7_matmul.py -k {matmul_variant}"

    # The profiled runs inherit this process's environment: set the stagger for the first run,
    # clear it for the baseline, and hand the caller's own settings back whatever happens.
    saved_stagger_env = {key: os.environ.get(key) for key in STAGGER_ENV_VARS}
    try:
        os.environ["TT_MM_STAGGER_TYPE"] = stagger_type
        os.environ["TT_MM_STAGGER_VALUE"] = str(STAGGER_CYCLES)
        run_device_profiler(command, output_logs_subdir_with_stagger)
        results_with_stagger = post_process_ops_log(output_logs_subdir_with_stagger, [duration_column])

        for key in STAGGER_ENV_VARS:
            os.environ.pop(key, None)
        run_device_profiler(command, output_logs_subdir)
        results_without_stagger = post_process_ops_log(output_logs_subdir, [duration_column])
    finally:
        for key, value in saved_stagger_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    duration_with = results_with_stagger[duration_column]
    duration_without = results_without_stagger[duration_column]
    logger.info(
        f"{matmul_variant} stagger type {stagger_type}: {duration_column} with stagger = {duration_with:.0f}, "
        f"without = {duration_without:.0f}"
    )
    assert duration_with - duration_without > MIN_STAGGER_DELTA_NS, (
        f"Stagger type {stagger_type} with {STAGGER_CYCLES} cycles should make {matmul_variant} visibly slower: "
        f"with stagger {duration_with:.0f} ns, without {duration_without:.0f} ns"
    )
