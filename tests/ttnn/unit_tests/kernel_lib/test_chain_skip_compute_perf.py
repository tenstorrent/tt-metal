# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-profiler A/B coverage for CKL_ELTWISE_CHAIN_SKIP_COMPUTE.

The profile fixtures run identical kernels with the macro off and on. They cover every ordinary
setup placement plus the managed/caller-managed and block-size variants of L1 and DEST
accumulation. The outer tests invoke each fixture through the device profiler and require the
skipped DEVICE KERNEL duration to decrease.
"""

import pytest
import ttnn
from loguru import logger

import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib
from models.perf.device_perf_utils import run_device_perf_detailed

pytestmark = pytest.mark.models_device_performance_bare_metal

OP = "GenericOpDeviceOperation"
PERF_FILE = "tests/ttnn/unit_tests/kernel_lib/test_chain_skip_compute_perf.py"
HOIST_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/axes/hoist.cpp"
ACCUMULATION_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/accumulation.cpp"
SKIP_DEFINE = [("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "1")]
N_ITERS = 20
CASES = {
    "ordinary-hoisted": ("ordinary", 0),
    "ordinary-per-call": ("ordinary", 1),
    "ordinary-caller-setup": ("ordinary", 2),
    "l1-managed": ("l1", False),
    "l1-caller-managed": ("l1", True),
    "dest-per-row-b1-managed": ("dest", False, 1, False),
    "dest-per-row-b1-caller-managed": ("dest", False, 1, True),
    "dest-per-row-b8-managed": ("dest", False, 8, False),
    "dest-per-row-b8-caller-managed": ("dest", False, 8, True),
    "dest-whole-shape-b1-managed": ("dest", True, 1, False),
    "dest-whole-shape-b1-caller-managed": ("dest", True, 1, True),
    "dest-whole-shape-b8-managed": ("dest", True, 8, False),
    "dest-whole-shape-b8-caller-managed": ("dest", True, 8, True),
}
VARIANTS = tuple(f"{case}-{mode}" for case in CASES for mode in ("run", "skip"))


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


def _l1(device, variant, caller_managed):
    n = 256
    dtype = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()
    _, tt_in = lib.make_input(shape, dtype, device, seed=91002, scale=0.125, bias=0.0)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32]), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    program = ttnn.ProgramDescriptor(
        kernels=[
            lib.build_reader_kernel([tt_in], n, core_grid),
            lib.build_writer_1out_kernel(tt_out, 1, core_grid),
            lib.build_compute_kernel(
                ACCUMULATION_KERNEL,
                [1, n, 1, int(caller_managed), 1, 0],
                core_grid,
                defines=_defines(variant),
            ),
        ],
        semaphores=[],
        cbs=[
            lib.cb_descriptor(0, dtype, 2, core_grid),
            lib.cb_descriptor(15, dtype, 1, core_grid),
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


@pytest.mark.parametrize("variant", VARIANTS)
def test_profile_fixture(device, variant):
    case = variant.rsplit("-", 1)[0]
    kind, *args = CASES[case]
    if kind == "ordinary":
        tensors, program = _ordinary(device, variant, *args)
    elif kind == "l1":
        tensors, program = _l1(device, variant, *args)
    else:
        tensors, program = _dest(device, variant, *args)

    for _ in range(N_ITERS):
        ttnn.generic_op(tensors, program)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)


def _device_kernel_ns(variant):
    results = run_device_perf_detailed(
        command=f'pytest "{PERF_FILE}::test_profile_fixture[variant={variant}]" -v',
        subdir=f"eltwise_skip_compute_{variant}",
        cols=["DEVICE KERNEL"],
        op_name=OP,
        warmup_iters=2,
    )
    return results["DEVICE KERNEL"]["AVG"]


@pytest.mark.parametrize("case", CASES)
def test_skip_compute_reduces_device_kernel_time(case):
    run_ns = _device_kernel_ns(f"{case}-run")
    skip_ns = _device_kernel_ns(f"{case}-skip")
    logger.info(
        f"skip-compute {case} | run={run_ns:.0f} ns | skip={skip_ns:.0f} ns | " f"speedup={run_ns / skip_ns:.3f}x"
    )
    assert skip_ns < run_ns, f"{case}: skip did not reduce DEVICE KERNEL time ({skip_ns:.0f} >= {run_ns:.0f} ns)"
