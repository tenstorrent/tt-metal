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
  fab_l1     - same_core, plus fabric: on every chip two port cores (BRISC, NOC1 - as the head's gather / RS ports)
               stream 8 KB packets to an X neighbour's same core L1 slots, a fused write + atomic increment every 4th
  fab_dram   - fab_l1 but the packets land in the neighbour's DRAM (the receiving router's NOC1 DRAM writes)
  mcast      - same_core, plus each column's first loader multicasting what it read down its column (NOC1, loopback:
               the W stream's shape)
  fab_mcast  - mcast plus fab_dram
  mcast_posted / mcast_nowrite - mcast with the writers' DRAM writes posted / gone (their reads kept)
A wedge shows as a dispatch timeout (run it under run_safe_pytest: triage on).

Results (bh-lb-09, 2x4, 200 programs each): remote / same_core / posted / same_fence / fab_l1 / fab_dram - no wedge
(the head's wedge needs more than this mix). Every mcast mode - even mcast_nowrite and mcast_only11_noreads - stalls
between program 150 and 200 on logical column 11's row 0 / 1 cores on every chip, also with no lead on column 11: a
probe artifact around column 11 (the head's movers column, never a W multicast column), not chased."""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.moe_block.tests.test_posted_fence import _l1

_WRITER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t UNITS = get_compile_time_arg_val(0), MODE = get_compile_time_arg_val(1), NL = get_compile_time_arg_val(2);
constexpr uint32_t NW = 32, SZ = 640, LB = 8192;  // MODE: 0 remote, 1 same_core, 2 posted, 3 same_fence, 4 no writes
void kernel_main() {
    uint32_t a = 0;
    const uint32_t data = get_arg_val<uint32_t>(a++), w = get_arg_val<uint32_t>(a++), buf = get_arg_val<uint32_t>(a++);
    const uint32_t lsrc = get_arg_val<uint32_t>(a++), ldst = get_arg_val<uint32_t>(a++), stop = get_arg_val<uint32_t>(a++);
    const uint32_t res = get_arg_val<uint32_t>(a++);
    uint32_t n = 0;
    for (uint32_t u = 0; u < UNITS; ++u) {
        if constexpr (MODE != 0 && MODE != 5) {  // (my own reads in flight while the unit's writes go)
            for (uint32_t q = 0; q < 3; ++q, ++n) {
                noc_async_read(get_noc_addr_from_bank_id<true>((w + n) % NUM_DRAM_BANKS, lsrc + (n % 8) * LB), ldst + q * LB, LB);
            }
        }
        for (uint32_t j = 0; j < NW; ++j) {
            const uint32_t bank = (w + j + u) % NUM_DRAM_BANKS, off = data + ((w * NW + j) % 256) * SZ;
            const uint64_t dst = get_noc_addr_from_bank_id<true>(bank, off);
            if constexpr (MODE >= 4) (void)dst;
            else if constexpr (MODE >= 2) noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(buf + j * SZ, dst, SZ);
            else noc_async_write(buf + j * SZ, dst, SZ);
        }
        if constexpr (MODE >= 2) noc_async_posted_writes_flushed();
        else noc_async_writes_flushed();
        if constexpr (MODE == 3) {
            for (uint32_t b = 0; b < NUM_DRAM_BANKS; ++b) noc_async_write(buf, get_noc_addr_from_bank_id<true>(b, data + w * 64), 4);
            noc_async_write_barrier();
        }
        if constexpr (MODE != 0 && MODE != 5) noc_async_read_barrier();
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
        const uint32_t mc = get_arg_val<uint32_t>(7);  // (column multicast lead: x | y0 << 8 | y1 << 16 | cores << 24)
        if (mc) {  // (the W stream's shape: what it read, down its column on NOC1, loopback)
            const uint32_t x = mc & 0xFF, y0 = (mc >> 8) & 0xFF, y1 = (mc >> 16) & 0xFF;
            for (uint32_t q = 0; q < 4; ++q) {
                noc_async_write_multicast_loopback_src(dst + q * sz, get_noc_multicast_addr(x, y1, x, y0, dst + 4 * sz + q * sz), sz, mc >> 24);  // (NOC1: from the far corner)
            }
            noc_async_write_barrier();
        }
        invalidate_l1_cache();
        if (*s >= nw) break;
    }
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(res)[0] = n;
}
"""

_SEND = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
using namespace tt::tt_fabric;
constexpr uint32_t NH = 6, TO_DRAM = get_compile_time_arg_val(0);
void kernel_main() {
    size_t a = 0;
    const uint32_t dst_chip = get_arg_val<uint32_t>(a++), mesh = get_arg_val<uint32_t>(a++);
    const uint32_t dx = get_arg_val<uint32_t>(a++), dy = get_arg_val<uint32_t>(a++), base = get_arg_val<uint32_t>(a++);
    const uint32_t slots = get_arg_val<uint32_t>(a++), n = get_arg_val<uint32_t>(a++), pkt = get_arg_val<uint32_t>(a++);
    const uint32_t done = get_arg_val<uint32_t>(a++), src = get_arg_val<uint32_t>(a++), dbase = get_arg_val<uint32_t>(a++);
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr[NH];
    for (uint32_t i = 0; i < NH; ++i) {
        hdr[i] = PacketHeaderPool::allocate_header();
        (void)fabric_set_unicast_route(hdr[i], static_cast<uint16_t>(dst_chip), static_cast<uint16_t>(mesh));
    }
    conn.open();
    for (uint32_t i = 0; i < n; ++i) {
        auto* h = hdr[i % NH];
        const uint64_t to = TO_DRAM ? get_noc_addr_from_bank_id<true>(i % NUM_DRAM_BANKS, dbase + (i % slots) * pkt)
                                 : get_noc_addr(dx, dy, base + (i % slots) * pkt);
        if (i % 4 == 3) {
            h->to_noc_fused_unicast_write_atomic_inc(
                NocUnicastAtomicIncFusedCommandHeader{to, get_noc_addr(dx, dy, done + 16), 1, true}, pkt);
        } else {
            h->to_noc_unicast_write(NocUnicastCommandHeader{to}, pkt);
        }
        conn.wait_for_empty_write_slot();
        conn.send_current_slot_non_blocking(src, pkt, reinterpret_cast<uint32_t>(h));
        if (i % NH == NH - 1) noc_async_writes_flushed();
    }
    noc_async_writes_flushed();
    hdr[0]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(dx, dy, done), 1, true});
    conn.wait_for_empty_write_slot();
    conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr[0]), sizeof(PACKET_HEADER_TYPE));
    conn.close();
    noc_async_atomic_barrier();
}
"""

_RECV = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    const uint32_t done = get_arg_val<uint32_t>(0), want = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done);
    while (true) {
        invalidate_l1_cache();
        if (d[0] >= want) break;
    }
}
"""

MODES = {
    "remote": 0,
    "same_core": 1,
    "posted": 2,
    "same_fence": 3,
    "fab_l1": 1,
    "fab_dram": 1,
    "mcast": 1,
    "fab_mcast": 1,
    "mcast_posted": 2,
    "mcast_nowrite": 4,
    "mcast_nowrite_no11": 4,  # (multicast leads on every column but 11)
    "mcast_nowrite_only11": 4,  # (column 11's lead only)
    "mcast_only11_noreads": 5,  # (column 11's lead only; the writers neither read nor write)
}
PORTS = ((11, 2), (11, 3))  # (the head's port cores; link 0 / 1)
PKT, NPKT = 8192, 20000


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((2, 4), fabric2d_device_params(fabric_payload_size=8256, l1_small_size=0))],
    indirect=True,
)
@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize(
    "units, programs", [(2000, 200)]
)  # (fabric modes: 20k 8 KB packets a port a program)  # (a program well under the dispatch timeout, many of them)
@pytest.mark.timeout(3600)
def test_noc1_traffic(mesh_device, device_params, mode, units, programs):
    grid = mesh_device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    writers = [c for c in cores if c.y == 0]
    fab = mode.startswith("fab")
    loaders = [c for c in cores if c.y > 0 and not (fab and (c.x, c.y) in PORTS)]
    LB, NB = 8192, 32 * 640 + 3 * 8192 + 256 + 4 * PKT  # (+ the fabric's L1 slots: 4 packets)
    l1 = _l1(mesh_device, crs, len(cores), NB + 8 * LB)
    base = int(l1.buffer_address())
    buf, ldst_w, words = base, base + 32 * 640, base + 32 * 640 + 3 * 8192
    fslots, ldst = words + 256, base + NB
    stop, res = words, words + 16
    ttnn.copy(
        ttnn.from_torch(
            torch.zeros(len(cores), (NB + 8 * LB) // 4, dtype=torch.int32),
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
    lead = {}
    if "mcast" in mode:  # (each column's first loader; its column's loaders - rows 1 .., ports excluded: they're in it)
        for c in loaders:
            if ("no11" in mode and c.x == 11) or ("only11" in mode and c.x != 11):
                continue
            lead.setdefault(c.x, c)
    for i, c in enumerate(loaders):
        mc = 0
        if lead.get(c.x) is c:
            col = [l for l in loaders if l.x == c.x]
            v0, v1 = virt(ttnn.CoreCoord(c.x, min(l.y for l in col))), virt(ttnn.CoreCoord(c.x, grid.y - 1))
            mc = v0.x | v0.y << 8 | v1.y << 16 | (grid.y - min(l.y for l in col)) << 24
        lrt[c.x][c.y] = [lsrc, ldst, stop, res, LB, i, len(writers), mc]
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
    if not fab:
        md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, C - 1))] = prog
    else:  # (every chip its own program: its port senders toward an X neighbour, receivers for who sends to it)
        node = lambda rc: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*rc))  # noqa: E731
        dst = {(r, c): (r, c + 1 if c + 1 < C else c - 1) for r in range(R) for c in range(C)}
        want = {}
        for s_, d_ in dst.items():
            want[d_] = want.get(d_, 0) + 1
        dfab = dbase + 384 * 1024  # (the DRAM slots: 4 packets a bank, past the loaders' sources)
        for rc in dst:
            p = ttnn.ProgramDescriptor()
            ks = list(prog.kernels)
            for l, xy in enumerate(PORTS):
                cc = ttnn.CoreCoord(*xy)
                v = virt(cc)
                dn = node(dst[rc])
                rt = ttnn.RuntimeArgs()
                rt[cc.x][cc.y] = [
                    int(dn.chip_id),
                    int(dn.mesh_id),
                    v.x,
                    v.y,
                    fslots,
                    4,
                    NPKT,
                    PKT,
                    stop + 32,
                    buf,
                    dfab,
                ] + list(ttnn.setup_fabric_connection(node(rc), dn, l, p, cc))
                ks.append(
                    ttnn.KernelDescriptor(
                        kernel_source=_SEND,
                        source_type=st,
                        core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(cc, cc)]),
                        compile_time_args=[int(mode == "fab_dram")],
                        runtime_args=rt,
                        config=ttnn.DataMovementConfigDescriptor(
                            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
                        ),
                    )
                )
                if rc in want:
                    rr = ttnn.RuntimeArgs()
                    rr[cc.x][cc.y] = [stop + 32, want[rc]]
                    ks.append(
                        ttnn.KernelDescriptor(
                            kernel_source=_RECV,
                            source_type=st,
                            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(cc, cc)]),
                            compile_time_args=[],
                            runtime_args=rr,
                            config=ttnn.DataMovementConfigDescriptor(  # (polls its L1 only: off the sender's NOC1)
                                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0
                            ),
                        )
                    )
            p.kernels = ks
            md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*rc), ttnn.MeshCoordinate(*rc))] = p
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
                torch.zeros(len(cores), (NB + 8 * LB) // 4, dtype=torch.int32),
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
