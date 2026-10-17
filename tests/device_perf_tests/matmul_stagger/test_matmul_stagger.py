# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import pytest
from loguru import logger

from models.common.utility_functions import is_blackhole, is_wormhole_b0
from tracy.process_model_log import post_process_ops_log, run_device_profiler
from tests.device_perf_tests.matmul_stagger.test_run_8x7_matmul import MATMUL_VARIANTS

# Stagger type 2 delays the odd-row cores by TT_MM_STAGGER_VALUE cycles before every in1 block
# (tt_metal/hw/ckernels/<arch>/metal/llk_io/llk_io_unpack.h). One million cycles is at least
# 500 us on any Tensix clock, so the op has to get slower by far more than its run-to-run noise.
STAGGER_TYPE = "2"
STAGGER_CYCLES = 1_000_000
MIN_STAGGER_DELTA_NS = 250_000


# for all matmul variants that can enable stagger, profile op with stagger enabled and disabled,
# and make sure the execution time of the op is longer with stagger enabled
@pytest.mark.skipif(
    not (is_wormhole_b0() or is_blackhole()), reason="matmul stagger is only enabled on Wormhole B0 and Blackhole"
)
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "matmul_variant",
    MATMUL_VARIANTS,
)
def test_matmul_stagger(matmul_variant):
    duration_column = "DEVICE KERNEL DURATION [ns]"
    output_logs_subdir = f"test_matmul_stagger_{matmul_variant}"

    command = f"pytest tests/device_perf_tests/matmul_stagger/test_run_8x7_matmul.py -k {matmul_variant}"

    os.environ["TT_MM_STAGGER_TYPE"] = STAGGER_TYPE
    os.environ["TT_MM_STAGGER_VALUE"] = str(STAGGER_CYCLES)
    output_logs_subdir_with_stagger = output_logs_subdir + "_with_stagger"
    try:
        run_device_profiler(command, output_logs_subdir_with_stagger)
    finally:
        del os.environ["TT_MM_STAGGER_TYPE"]
        del os.environ["TT_MM_STAGGER_VALUE"]
    results_with_stagger = post_process_ops_log(output_logs_subdir_with_stagger, [duration_column])

    run_device_profiler(command, output_logs_subdir)
    results_without_stagger = post_process_ops_log(output_logs_subdir, [duration_column])

    duration_with = results_with_stagger[duration_column]
    duration_without = results_without_stagger[duration_column]
    logger.info(
        f"{matmul_variant}: {duration_column} with stagger = {duration_with:.0f}, without = {duration_without:.0f}"
    )
    assert duration_with - duration_without > MIN_STAGGER_DELTA_NS, (
        f"Stagger type {STAGGER_TYPE} with {STAGGER_CYCLES} cycles should make {matmul_variant} visibly slower: "
        f"with stagger {duration_with:.0f} ns, without {duration_without:.0f} ns"
    )
