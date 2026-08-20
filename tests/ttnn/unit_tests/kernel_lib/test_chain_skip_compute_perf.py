# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Real-time-profiler A/B coverage for CKL_ELTWISE_CHAIN_SKIP_COMPUTE.

The two representative fixtures run identical kernels with the macro off and on. The outer tests
measure each program in-process, assert its recorded Wormhole RT baseline, and require the skipped
device-program duration to decrease. The remaining skip configurations have functional coverage in
test_chain_skip_compute.py.
"""

import statistics

import pytest
import ttnn
from loguru import logger

from models.common.utility_functions import is_wormhole_b0
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib
from tests.ttnn.profiling.realtime_profiler_utils import collect_op_durations_merged, require_realtime_profiler

pytestmark = pytest.mark.models_device_performance_bare_metal

HOIST_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/axes/hoist.cpp"
ACCUMULATION_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/accumulation.cpp"
SKIP_DEFINE = [("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "1")]
N_ITERS = 20
CASES = {
    "ordinary-per-call": ("ordinary", 1),
    "dest-per-row-b1-managed": ("dest", False, 1, False),
}
RT_BASELINE_MARGIN = 0.02
wormhole_rt_baseline = pytest.mark.skipif(not is_wormhole_b0(), reason="RT baselines are recorded on Wormhole B0")
RT_BASELINE_NS = {
    "ordinary-per-call": {"run": 193132, "skip": 126922},
    "dest-per-row-b1-managed": {"run": 47149, "skip": 41490},
}


def _defines(variant):
    return SKIP_DEFINE if variant.endswith("-skip") else None


def _ordinary(device, variant, setup_mode):
    n = 256
    dtype = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()
    _, tt_in = lib.make_input(shape, dtype, device, seed=91001)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    program = ttnn.ProgramDescriptor(
        kernels=[
            lib.build_reader_kernel([tt_in], n, core_grid),
            lib.build_writer_1out_kernel(tt_out, n, core_grid),
            lib.build_compute_kernel(HOIST_KERNEL, [n, setup_mode], core_grid, defines=_defines(variant)),
        ],
        semaphores=[],
        cbs=[
            lib.cb_descriptor(0, dtype, 2, core_grid),
            lib.cb_descriptor(16, dtype, 2, core_grid),
        ],
    )
    return [tt_in, tt_out], program


def _dest(device, variant, whole_shape, block_size, caller_managed):
    tiles_per_output = 8
    num_outputs = 8
    total_input_tiles = tiles_per_output * num_outputs
    output_tiles = 1 if whole_shape else num_outputs
    dtype = ttnn.bfloat16
    core_grid = lib.single_core_grid()
    shape = [1, 1, 32, 32 * total_input_tiles]
    _, tt_a = lib.make_input(shape, dtype, device, seed=91003)
    _, tt_b = lib.make_input(shape, dtype, device, seed=91004)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32 * output_tiles]),
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    program = ttnn.ProgramDescriptor(
        kernels=[
            lib.build_reader_kernel([tt_a, tt_b], total_input_tiles, core_grid),
            lib.build_writer_1out_kernel(tt_out, output_tiles, core_grid),
            lib.build_compute_kernel(
                ACCUMULATION_KERNEL,
                [0, tiles_per_output, block_size, int(caller_managed), num_outputs, int(whole_shape)],
                core_grid,
                defines=_defines(variant),
            ),
        ],
        semaphores=[],
        cbs=[
            lib.cb_descriptor(0, dtype, total_input_tiles, core_grid),
            lib.cb_descriptor(1, dtype, total_input_tiles, core_grid),
            lib.cb_descriptor(16, dtype, output_tiles, core_grid),
        ],
    )
    return [tt_a, tt_b, tt_out], program


def _build_variant(device, variant):
    case = variant.rsplit("-", 1)[0]
    kind, *args = CASES[case]
    if kind == "ordinary":
        return _ordinary(device, variant, *args), HOIST_KERNEL
    return _dest(device, variant, *args), ACCUMULATION_KERNEL


def _realtime_program_ns(device, variant):
    """Median RT duration of freshly dispatched GenericOp programs."""
    require_realtime_profiler("eltwise-chain skip-compute performance coverage")
    (tensors, program), kernel_path = _build_variant(device, variant)

    # RT callback subscriptions can replay older records, so retain only the newest measured
    # dispatches. Do not warm up: the RT record already excludes host-side compilation.
    durations = collect_op_durations_merged(
        device,
        lambda: ttnn.generic_op(tensors, program),
        kernel_path,
        iters=N_ITERS,
        allow_stale_prefix=True,
    )
    return statistics.median(durations)


@wormhole_rt_baseline
@pytest.mark.parametrize("case", CASES)
def test_skip_compute_reduces_device_program_time(device, case):
    run_ns = _realtime_program_ns(device, f"{case}-run")
    skip_ns = _realtime_program_ns(device, f"{case}-skip")
    logger.info(
        f"skip-compute {case} | RT program run={run_ns:.0f} ns | skip={skip_ns:.0f} ns | "
        f"speedup={run_ns / skip_ns:.3f}x"
    )
    for variant, measured_ns in (("run", run_ns), ("skip", skip_ns)):
        baseline_ns = RT_BASELINE_NS[case][variant]
        lower = baseline_ns * (1 - RT_BASELINE_MARGIN)
        upper = baseline_ns * (1 + RT_BASELINE_MARGIN)
        assert lower <= measured_ns <= upper, (
            f"{case} {variant}: {measured_ns:.0f} ns outside the RT baseline {baseline_ns} ± "
            f"{RT_BASELINE_MARGIN * 100:.0f}% ({lower:.0f}-{upper:.0f} ns)"
        )
    assert skip_ns < run_ns, f"{case}: skip did not reduce device-program time ({skip_ns:.0f} >= {run_ns:.0f} ns)"
