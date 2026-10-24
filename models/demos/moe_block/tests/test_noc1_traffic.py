# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Traffic-only probe (no semaphores between cores, no CBs, no compute) for the NOC1 wedge the fused head hit with its
prep's DRAM row writes under its W stream (MOE_BLOCK_WORKLOG.md): which NOC1 traffic mix wedges a chip.

Writer cores (row 0, NCRISC, NOC1) issue units of 32 x 640 B DRAM writes (the prep's shape), loader cores (the rest,
NCRISC, NOC1) stream 8 KB DRAM reads, four in flight, until the writers are done. mode:
  remote     - writers only write (non-posted, each unit flushed): the reads are other cores'
  same_core  - writers also keep their own NOC1 DRAM reads in flight between the units (the engine NCRISC's W stream)
  posted     - same_core with posted writes
  same_fence - same_core with posted writes and, after each unit, a non-posted 4 B write a bank then the barrier
A wedge shows as a dispatch timeout (run it under run_safe_pytest: triage on)."""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.moe_block.tests.test_posted_fence import _l1

_WRITER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t UNITS = get_compile_time_arg_val(0), MODE = get_compile_time_arg_val(1), NL = get_compile_time_arg_val(2);
constexpr uint32_t NW = 32, SZ = 640, LB = 8192;  // MODE: 0 remote, 1 same_core, 2 posted, 3 same_fence
void kernel_main() {
    uint32_t a = 0;
    const uint32_t data = get_arg_val<uint32_t>(a++), w = get_arg_val<uint32_t>(a++), buf = get_arg_val<uint32_t>(a++);
    const uint32_t lsrc = get_arg_val<uint32_t>(a++), ldst = get_arg_val<uint32_t>(a++), stop = get_arg_val<uint32_t>(a++);
    const uint32_t res = get_arg_val<uint32_t>(a++);
    uint32_t n = 0;
    for (uint32_t u = 0; u < UNITS; ++u) {
        if constexpr (MODE != 0) {  // (my own reads in flight while the unit's writes go)
            for (uint32_t q = 0; q < 3; ++q, ++n) {
                noc_async_read(get_noc_addr_from_bank_id<true>((w + n) % NUM_DRAM_BANKS, lsrc + (n % 8) * LB), ldst + q * LB, LB);
            }
        }
        for (uint32_t j = 0; j < NW; ++j) {
            const uint32_t bank = (w + j + u) % NUM_DRAM_BANKS, off = data + ((w * NW + j) % 256) * SZ;
            const uint64_t dst = get_noc_addr_from_bank_id<true>(bank, off);
            if constexpr (MODE >= 2) noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(buf + j * SZ, dst, SZ);
            else noc_async_write(buf + j * SZ, dst, SZ);
        }
        if constexpr (MODE >= 2) noc_async_posted_writes_flushed();
        else noc_async_writes_flushed();
        if constexpr (MODE == 3) {
            for (uint32_t b = 0; b < NUM_DRAM_BANKS; ++b) noc_async_write(buf, get_noc_addr_from_bank_id<true>(b, data + w * 64), 4);
            noc_async_write_barrier();
        }
        if constexpr (MODE != 0) noc_async_read_barrier();
    }
    noc_async_write_barrier();
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(res)[0] = UNITS;
    for (uint32_t l = 0; l < NL; ++l) {
        const uint32_t xy = get_arg_val<uint32_t>(a + l);
        noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, stop), 1);
    }
    noc_async_atomic_barrier();
}
"""

_LOADER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    const uint32_t src = get_arg_val<uint32_t>(0), dst = get_arg_val<uint32_t>(1), stop = get_arg_val<uint32_t>(2);
    const uint32_t res = get_arg_val<uint32_t>(3), sz = get_arg_val<uint32_t>(4), id = get_arg_val<uint32_t>(5);
    const uint32_t nw = get_arg_val<uint32_t>(6);  // (the writers: stop once every one has)
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop);
    uint32_t n = 0;
    while (true) {
        for (uint32_t q = 0; q < 4; ++q, ++n) {
            noc_async_read(get_noc_addr_from_bank_id<true>((id + n) % NUM_DRAM_BANKS, src + (n % 8) * sz), dst + q * sz, sz);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        if (*s >= nw) break;
    }
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(res)[0] = n;
}
"""

MODES = {"remote": 0, "same_core": 1, "posted": 2, "same_fence": 3}


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("units, programs", [(2000, 200)])  # (a program well under the dispatch timeout, many of them)
@pytest.mark.timeout(3600)
def test_noc1_traffic(mesh_device, mode, units, programs):
    grid = mesh_device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    writers = [c for c in cores if c.y == 0]
    loaders = [c for c in cores if c.y > 0]
    LB, NB = 8192, 32 * 640 + 3 * 8192 + 256
    l1 = _l1(mesh_device, crs, len(cores), NB + 4 * LB)
    base = int(l1.buffer_address())
    buf, ldst_w, words, ldst = base, base + 32 * 640, base + 32 * 640 + 3 * 8192, base + NB
    stop, res = words, words + 16
    ttnn.copy(
        ttnn.from_torch(
            torch.zeros(len(cores), (NB + 4 * LB) // 4, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=l1.memory_config(),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
        l1,
    )
    nbank = mesh_device.dram_grid_size().x
    dram = ttnn.allocate_tensor_on_device(
        ttnn.Shape([nbank * 64, 2048]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    dbase = int(dram.buffer_address())
    data, lsrc = dbase, dbase + 256 * 1024
    virt = lambda c: mesh_device.worker_core_from_logical_core(c)  # noqa: E731
    lxy = [virt(c).x | virt(c).y << 16 for c in loaders]
    wrt, lrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for w, c in enumerate(writers):
        wrt[c.x][c.y] = [data, w, buf, lsrc, ldst_w, stop, res] + lxy
    for i, c in enumerate(loaders):
        lrt[c.x][c.y] = [lsrc, ldst, stop, res, LB, i, len(writers)]
    st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    nc = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default)
    prog = ttnn.ProgramDescriptor()
    prog.kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_WRITER,
            source_type=st,
            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in writers]),
            compile_time_args=[units, MODES[mode], len(loaders)],
            runtime_args=wrt,
            config=nc,
        ),
        ttnn.KernelDescriptor(
            kernel_source=_LOADER,
            source_type=st,
            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in loaders]),
            compile_time_args=[],
            runtime_args=lrt,
            config=nc,
        ),
    ]
    R, C = tuple(mesh_device.shape)
    md = ttnn.MeshProgramDescriptor()
    md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, C - 1))] = prog
    import time

    for k in range(programs):
        t0 = time.perf_counter()
        ttnn.generic_op([l1, dram], md)
        ttnn.synchronize_device(mesh_device)
        if k < 2 or k % 50 == 49:
            logger.info(
                f"NOC1_TRAFFIC {mode}: program {k + 1} of {programs}, {(time.perf_counter() - t0) * 1e3:.0f} ms"
            )
        ttnn.copy(
            ttnn.from_torch(
                torch.zeros(len(cores), (NB + 4 * LB) // 4, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=l1.memory_config(),
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
            l1,
        ) if k + 1 < programs else None
    for d, t in enumerate(ttnn.get_device_tensors(l1)):
        v = ttnn.to_torch(t).to(torch.int64).reshape(len(cores), -1) & 0xFFFFFFFF
        assert all(v[cores.index(c), (res - base) // 4] == units for c in writers)
    logger.info(
        f"NOC1_TRAFFIC {mode}: {programs} programs x {units} units x {len(writers)} writers x 8 chips - no wedge"
    )
