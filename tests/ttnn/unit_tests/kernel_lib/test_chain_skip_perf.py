# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-kernel-time A/B driver for the CKL_ELTWISE_CHAIN_SKIP_COMPUTE knob.

Each parametrized case just runs ONE fixture (Run macro-off or Skip macro-on) N times on device.
Run this under the tracy wrapper, one case per invocation, so each produces its own device profiler
CSV; the per-RISC *-KERNEL durations are parsed and compared out-of-band. Skip keeps the whole CB
lifecycle + tile_regs window and elides only compute, so the compute engines (TRISC_0/1/2) must
collapse vs Run while the data-movement RISCs (BRISC/NCRISC) stay — the real proof, beyond PCC."""

import pytest
import ttnn
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

N_ITERS = 20

KERNELS = {
    "ord_run": "ttnn/cpp/ttnn/kernel_lib/tests/axes/hoist_single_call.cpp",
    "ord_skip": "ttnn/cpp/ttnn/kernel_lib/tests/axes/skip_compute_exp.cpp",
    "dest_run": "ttnn/cpp/ttnn/kernel_lib/tests/dest_accumulation/dest_accumulation.cpp",
    "dest_skip": "ttnn/cpp/ttnn/kernel_lib/tests/dest_accumulation/skip_compute_dest_accum.cpp",
    "l1_run": "ttnn/cpp/ttnn/kernel_lib/tests/l1_accumulation/l1_accumulation.cpp",
    "l1_skip": "ttnn/cpp/ttnn/kernel_lib/tests/l1_accumulation/skip_compute_l1_accum.cpp",
}


def _ordinary(device, kernel):
    n = 64
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    cg = lib.single_core_grid()
    _, tt_in = lib.make_input(shape, dt, device, seed=777)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [lib.cb_descriptor(0, dt, 2, cg), lib.cb_descriptor(16, dt, 2, cg)]
    reader = lib.build_reader_kernel([tt_in], n, cg)
    writer = lib.build_writer_1out_kernel(tt_out, n, cg)
    compute = lib.build_compute_kernel(kernel, [n], cg)
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    return lambda: ttnn.generic_op([tt_in, tt_out], prog)


def _dest(device, kernel):
    n, num_outputs, block_size = 8, 8, 1
    total = n * num_outputs
    dt = ttnn.bfloat16
    cg = lib.single_core_grid()
    _, tt_local = lib.make_input([1, 1, 32, 32 * total], dt, device, seed=1701)
    _, tt_remote = lib.make_input([1, 1, 32, 32 * total], dt, device, seed=1702)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32 * num_outputs]), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    cbs = [
        lib.cb_descriptor(0, dt, total, cg),
        lib.cb_descriptor(1, dt, total, cg),
        lib.cb_descriptor(16, dt, num_outputs, cg),
    ]
    reader = lib.build_reader_kernel([tt_local, tt_remote], total, cg)
    writer = lib.build_writer_1out_kernel(tt_out, num_outputs, cg)
    compute = lib.build_compute_kernel(kernel, [n, block_size, 0, num_outputs], cg)
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    return lambda: ttnn.generic_op([tt_local, tt_remote, tt_out], prog)


def _l1(device, kernel):
    n = 64
    dt = ttnn.bfloat16
    cg = lib.single_core_grid()
    _, tt_in = lib.make_input([1, 1, 32, 32 * n], dt, device, seed=1701, scale=0.125, bias=0.0)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32]), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    cbs = [lib.cb_descriptor(0, dt, 2, cg), lib.cb_descriptor(15, dt, 1, cg), lib.cb_descriptor(16, dt, 2, cg)]
    reader = lib.build_reader_kernel([tt_in], n, cg)
    writer = lib.build_writer_1out_kernel(tt_out, 1, cg)
    compute = lib.build_compute_kernel(kernel, [n, 0], cg)
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    return lambda: ttnn.generic_op([tt_in, tt_out], prog)


_BUILDERS = {"ord": _ordinary, "dest": _dest, "l1": _l1}


@pytest.mark.parametrize("variant", list(KERNELS))
def test_skip_perf_variant(device, variant):
    walk = variant.rsplit("_", 1)[0]
    run_fn = _BUILDERS[walk](device, KERNELS[variant])
    for _ in range(N_ITERS):
        run_fn()
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)  # flush device-side *-KERNEL zones to profile_log_device.csv
