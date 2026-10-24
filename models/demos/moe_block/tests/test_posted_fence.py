# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The ordering a posted-write fence relies on, tested on the device: a core's posted DRAM writes (NCRISC, NOC1), then
one non-posted 4 B write to each DRAM bank it wrote, on the same VC, then the write barrier - once that ack is back, the
posted data must be in DRAM (the same source / destination pair and VC deliver in order). A checker (the same core's
BRISC, reading on NOC0) reads the data back right after each fence and compares every word, while every other core's
NCRISC streams NOC1 DRAM reads (the load the fused head's W stream puts on NOC1). fence=False (posted writes, only
flushed out of L1) is the control: it shows the checker can catch data not yet landed.

(MOE_BLOCK_WORKLOG.md: non-posted DRAM writes on NOC1 under NOC1 DRAM read load wedged a chip; the prep's writes went
posted with this fence.)"""

import pytest
import torch
from loguru import logger

import ttnn

NW, SZ = 32, 640  # (a prep unit: 32 row pieces of 640 B)

_WRITER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t ITERS = get_compile_time_arg_val(0), FENCE = get_compile_time_arg_val(1), NL = get_compile_time_arg_val(2);
constexpr uint32_t NW = __NW__, SZ = __SZ__;
void kernel_main() {
    uint32_t a = 0;
    const uint32_t data = get_arg_val<uint32_t>(a++), fence = get_arg_val<uint32_t>(a++), w = get_arg_val<uint32_t>(a++);
    const uint32_t buf = get_arg_val<uint32_t>(a++), flag = get_arg_val<uint32_t>(a++), ack = get_arg_val<uint32_t>(a++);
    const uint32_t stop = get_arg_val<uint32_t>(a++), one = get_arg_val<uint32_t>(a++);
    volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag);
    volatile tt_l1_ptr uint32_t* k = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ack);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(one) = 1;
    for (uint32_t it = 0; it < ITERS; ++it) {
        uint32_t* p = reinterpret_cast<uint32_t*>(buf);
        for (uint32_t j = 0; j < NW; ++j) {
            for (uint32_t i = 0; i < SZ / 4; ++i) p[j * SZ / 4 + i] = (it << 16) ^ (w << 11) ^ (j << 6) ^ i;
        }
        for (uint32_t j = 0; j < NW; ++j) {
            const uint32_t bank = (w + j) % NUM_DRAM_BANKS, off = data + (w * NW + j) * SZ;
            noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(buf + j * SZ, get_noc_addr_from_bank_id<true>(bank, off), SZ);
        }
        if constexpr (FENCE) {  // (one non-posted 4 B write a bank, same VC, then its ack)
            for (uint32_t b = 0; b < NUM_DRAM_BANKS; ++b) {
                noc_async_write(one, get_noc_addr_from_bank_id<true>(b, fence + w * 64), 4);
            }
            noc_async_write_barrier();
        } else {
            noc_async_posted_writes_flushed();
        }
        *f = it + 1;
        while (*k < it + 1) invalidate_l1_cache();
    }
    for (uint32_t l = 0; l < NL; ++l) {  // (the loaders: stop)
        const uint32_t xy = get_arg_val<uint32_t>(a + l);
        noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, stop), 1);
    }
    noc_async_atomic_barrier();
}
"""

_CHECKER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t ITERS = get_compile_time_arg_val(0);
constexpr uint32_t NW = __NW__, SZ = __SZ__;
void kernel_main() {
    uint32_t a = 0;
    const uint32_t data = get_arg_val<uint32_t>(a++), w = get_arg_val<uint32_t>(a++), chk = get_arg_val<uint32_t>(a++);
    const uint32_t flag = get_arg_val<uint32_t>(a++), ack = get_arg_val<uint32_t>(a++), res = get_arg_val<uint32_t>(a++);
    volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag);
    volatile tt_l1_ptr uint32_t* k = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ack);
    volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(res);
    uint32_t bad = 0, badit = 0, first = 0xFFFFFFFF;
    for (uint32_t it = 0; it < ITERS; ++it) {
        while (*f < it + 1) invalidate_l1_cache();
        for (uint32_t j = 0; j < NW; ++j) {
            const uint32_t bank = (w + j) % NUM_DRAM_BANKS, off = data + (w * NW + j) * SZ;
            noc_async_read(get_noc_addr_from_bank_id<true>(bank, off), chk + j * SZ, SZ);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        const uint32_t* c = reinterpret_cast<const uint32_t*>(chk);
        uint32_t b0 = bad;
        for (uint32_t j = 0; j < NW; ++j) {
            for (uint32_t i = 0; i < SZ / 4; ++i) bad += c[j * SZ / 4 + i] != ((it << 16) ^ (w << 11) ^ (j << 6) ^ i);
        }
        if (bad != b0) {
            ++badit;
            if (first == 0xFFFFFFFF) first = it;
        }
        *k = it + 1;
    }
    r[0] = ITERS;
    r[1] = bad;
    r[2] = badit;
    r[3] = first;
}
"""

_LOADER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    const uint32_t src = get_arg_val<uint32_t>(0), dst = get_arg_val<uint32_t>(1), stop = get_arg_val<uint32_t>(2);
    const uint32_t res = get_arg_val<uint32_t>(3), sz = get_arg_val<uint32_t>(4), id = get_arg_val<uint32_t>(5);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop);
    uint32_t n = 0;
    while (true) {
        for (uint32_t q = 0; q < 4; ++q, ++n) {  // (four reads in flight, every bank in turn)
            noc_async_read(get_noc_addr_from_bank_id<true>((id + n) % NUM_DRAM_BANKS, src + (n % 8) * sz), dst + q * sz, sz);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        if (*s) break;
    }
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(res)[0] = n;
}
"""


def _l1(mesh, crs, ncores, nbytes):
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([ncores, nbytes // 4]),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs, [1, nbytes // 4], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("fence", [True, False], ids=["fence", "control"])
@pytest.mark.parametrize("iters", [20000])
@pytest.mark.timeout(1800)
def test_posted_fence(mesh_device, fence, iters):
    grid = mesh_device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    writers = [c for c in cores if c.y == 0]
    loaders = [c for c in cores if c.y > 0]
    # L1 a core: buf (the pattern, NW SZ), chk (the read-back, NW SZ), words (flag, ack, stop, one, res x 4), load dst
    LB = 8192
    l1 = _l1(mesh_device, crs, len(cores), 2 * NW * SZ + 256 + 4 * LB)
    base = int(l1.buffer_address())
    buf, chk, words, ldst = base, base + NW * SZ, base + 2 * NW * SZ, base + 2 * NW * SZ + 256
    flag, ack, stop, one, res = words, words + 16, words + 32, words + 48, words + 64
    nbank = mesh_device.dram_grid_size().x
    dram = ttnn.allocate_tensor_on_device(
        ttnn.Shape([nbank * 64, 2048]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )  # (512 KB a bank: the writers' data, their fences, the loaders' sources)
    dbase = int(dram.buffer_address())
    data, fence_a, lsrc = dbase, dbase + len(writers) * NW * SZ, dbase + 256 * 1024
    zero = ttnn.from_torch(
        torch.zeros(len(cores), (2 * NW * SZ + 256 + 4 * LB) // 4, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=l1.memory_config(),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.copy(zero, l1)
    virt = lambda c: mesh_device.worker_core_from_logical_core(c)  # noqa: E731
    lxy = [virt(c).x | virt(c).y << 16 for c in loaders]
    wrt, crt, lrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for w, c in enumerate(writers):
        wrt[c.x][c.y] = [data, fence_a, w, buf, flag, ack, stop, one] + (lxy if w == 0 else [])
        crt[c.x][c.y] = [data, w, chk, flag, ack, res]
    for i, c in enumerate(loaders):
        lrt[c.x][c.y] = [lsrc, ldst, stop, res, LB, i]
    src = lambda s: s.replace("__NW__", str(NW)).replace("__SZ__", str(SZ))  # noqa: E731
    wcrs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in writers])
    lcrs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in loaders])
    st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    nc = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default)
    bc = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default)
    prog = ttnn.ProgramDescriptor()
    prog.kernels = [
        ttnn.KernelDescriptor(
            kernel_source=src(_WRITER),
            source_type=st,
            core_ranges=wcrs,
            compile_time_args=[iters, int(fence), len(loaders)],
            runtime_args=wrt,
            config=nc,
        ),
        ttnn.KernelDescriptor(
            kernel_source=src(_CHECKER), source_type=st, core_ranges=wcrs, compile_time_args=[iters], runtime_args=crt,
            config=bc,
        ),
        ttnn.KernelDescriptor(
            kernel_source=_LOADER, source_type=st, core_ranges=lcrs, compile_time_args=[], runtime_args=lrt, config=nc
        ),
    ]  # fmt: skip
    R, C = tuple(mesh_device.shape)
    md = ttnn.MeshProgramDescriptor()
    md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, C - 1))] = prog
    ttnn.generic_op([l1, dram], md)
    ttnn.synchronize_device(mesh_device)
    tot_bad = tot_it = 0
    for d, t in enumerate(ttnn.get_device_tensors(l1)):
        v = ttnn.to_torch(t).to(torch.int64).reshape(len(cores), -1) & 0xFFFFFFFF
        for w, c in enumerate(writers):
            r = v[cores.index(c), (res - base) // 4 : (res - base) // 4 + 4].tolist()
            assert r[0] == iters
            tot_bad += r[2]
            tot_it += r[0]
            if r[1]:
                logger.info(f"FENCE dev {d} writer {w}: {r[1]} bad words in {r[2]} iterations, first {r[3]}")
        loads = [v[cores.index(c), (res - base) // 4] for c in loaders]
        logger.info(f"FENCE dev {d}: loader reads {min(loads)} - {max(loads)} a core (8 KB each, NOC1)")
    logger.info(f"FENCE {'fence' if fence else 'control'}: {tot_bad} of {tot_it} writer iterations read stale data")
    if fence:
        assert tot_bad == 0, f"posted data not in DRAM after the fence's ack: {tot_bad} iterations"
