# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fabric link-rate probe on a mesh row (the TP line of a 2 x 4 LoudBox): what one fabric connection carries, per
link and direction, at the model's packet payload, FABRIC_2D vs FABRIC_1D, and what on the sender side limits it.

Per chip of row 0, a port core per (direction, link) (the placement of combine_rf.fabric_all_gather_py.plan) sends N
packets one hop to the neighbour's port core facing it; every INC-th packet is a fused write + increment of the
receiver's counter, whose kernel waits for all of them. Each sender stamps the wall clock around its send loop; rate =
N x packet bytes / (end - start), one link, one direction.

Environment (scratch probe):
  FLR_FABRIC   2d | 1d                              default 2d
  FLR_PAYLOAD  fabric max payload (router config)   default 8256 (GLM-5.3-Flash: 2 x 4096 + 64)
  FLR_PKT      bytes a packet (<= payload)          default the payload rounded down to 64
  FLR_N        packets a sender                     default 4096
  FLR_LINKS    1 | 2                                default 2
  FLR_DIRS     fwd | both                           default both
  FLR_SRC      l1 | dram                            default l1 (dram: a reader on NCRISC fills a CB from DRAM)
  FLR_GROUP    dram reads a batch (one barrier)     default 4
  FLR_DEPTH    CB depth in packets (dram)           default 8
  FLR_DST      l1 | dram                            default l1 (dram: the receiver lands packets in a DRAM buffer)
  FLR_HOLD     1: the sender re-uses one header a packet (route set once), 0: route written per packet  default 1
"""

import os
import statistics

import pytest
import torch
from loguru import logger
from ttnn.bringup.combine_rf.fabric_all_gather_py import NOC0, NOC1, allowed_cores, plan

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, fabric_1d_device_params

FABRIC = os.environ.get("FLR_FABRIC", "2d")
PAYLOAD = int(os.environ.get("FLR_PAYLOAD", "8256"))
PKT = int(os.environ.get("FLR_PKT", "0")) or PAYLOAD // 64 * 64
N = int(os.environ.get("FLR_N", "4096"))
LINKS = int(os.environ.get("FLR_LINKS", "2"))
DIRS = os.environ.get("FLR_DIRS", "both")
SRC = os.environ.get("FLR_SRC", "l1")
GROUP = int(os.environ.get("FLR_GROUP", "4"))
DEPTH = int(os.environ.get("FLR_DEPTH", "8"))
DST = os.environ.get("FLR_DST", "l1")
INC = 8
SLOTS = 8  # receiver ring (l1 destination)
AICLK_MHZ = 1350.0
_OPP = {"fwd": "bwd", "bwd": "fwd"}

_PARAMS = (fabric_1d_device_params if FABRIC == "1d" else fabric2d_device_params)(
    fabric_payload_size=PAYLOAD, l1_small_size=0
)

_SENDER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
constexpr uint32_t PKT = get_compile_time_arg_val(0), N = get_compile_time_arg_val(1), INC = get_compile_time_arg_val(2);
constexpr uint32_t SRC_DRAM = get_compile_time_arg_val(3), DST_DRAM = get_compile_time_arg_val(4);
constexpr uint32_t SLOTS = get_compile_time_arg_val(5), NH = get_compile_time_arg_val(6);
constexpr auto d_args = TensorAccessorArgs<7>();
FORCE_INLINE uint32_t clk() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }
void kernel_main() {
    size_t a = 0;
    const uint32_t res = get_arg_val<uint32_t>(a++), rx = get_arg_val<uint32_t>(a++), ry = get_arg_val<uint32_t>(a++);
    const uint32_t rbuf = get_arg_val<uint32_t>(a++), rctr = get_arg_val<uint32_t>(a++);
    const uint32_t dbuf = get_arg_val<uint32_t>(a++), page0 = get_arg_val<uint32_t>(a++);
    const uint16_t mesh = get_arg_val<uint32_t>(a++), chip = get_arg_val<uint32_t>(a++);
    const uint32_t src_l1 = get_arg_val<uint32_t>(a++);
    const auto dst = TensorAccessor(d_args, dbuf, PKT);
    uint32_t nslots = 0, bsize = 0;
    {  // (the worker channel's slots, from the L1 connection table device-init filled)
        tt_l1_ptr tensix_fabric_connections_l1_info_t* ci =
            reinterpret_cast<tt_l1_ptr tensix_fabric_connections_l1_info_t*>(FABRIC_CONNECTIONS_BASE);
        const uint32_t ch = get_arg_val<uint32_t>(a);
        nslots = ci->read_only[ch].num_buffers_per_channel;
        bsize = ci->read_only[ch].buffer_size_bytes;
    }
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[NH];
    for (uint32_t h = 0; h < NH; ++h) {
        hdrs[h] = PacketHeaderPool::allocate_header();
#if defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)
        (void)fabric_set_unicast_route(hdrs[h], chip, mesh);
#else
        (void)mesh;
        (void)chip;
        (void)fabric_set_unicast_route<false>(
            reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(hdrs[h]), static_cast<uint16_t>(1));
#endif
    }
    conn.open();
    const uint64_t ctr_noc = get_noc_addr(rx, ry, rctr);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(res);
    uint32_t stall = 0, h = 0, popped = 0;
    const uint32_t t0 = clk();
    for (uint32_t j = 0; j < N; ++j) {
        uint32_t src = src_l1;
        if constexpr (SRC_DRAM) {
            if (!cb_pages_available_at_front(0, popped + 1)) {
                const uint32_t t = clk();
                cb_wait_front(0, popped + 1);
                stall += clk() - t;
            }
            src = get_read_ptr(0) + popped * PKT;
        }
        const uint64_t d = DST_DRAM ? dst.get_noc_addr(page0 + j, 0, 0) : get_noc_addr(rx, ry, rbuf + (j % SLOTS) * PKT);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
        h = h + 1 == NH ? 0 : h + 1;
        if ((j + 1) % INC == 0 || j + 1 == N) {
            hdr->to_noc_fused_unicast_write_atomic_inc(NocUnicastAtomicIncFusedCommandHeader{d, ctr_noc, 1, true}, PKT);
        } else {
            hdr->to_noc_unicast_write(NocUnicastCommandHeader{d}, PKT);
        }
        if (!conn.edm_has_space_for_packet()) {
            const uint32_t t = clk();
            conn.wait_for_empty_write_slot();
            stall += clk() - t;
        }
        conn.send_current_slot_non_blocking(src, PKT, reinterpret_cast<uint32_t>(hdr));
        if constexpr (SRC_DRAM) {
            ++popped;
            if (popped == NH) {  // (a CB page and its header are free once their packet left L1)
                noc_async_writes_flushed();
                cb_pop_front(0, popped);
                popped = 0;
            }
        } else if (h == 0) {
            noc_async_writes_flushed();
        }
    }
    noc_async_writes_flushed();
    const uint32_t t1 = clk();
    if constexpr (SRC_DRAM) {
        if (popped) cb_pop_front(0, popped);
    }
    conn.close();
    st[0] = t0;
    st[1] = t1;
    st[2] = stall;
    st[3] = nslots;
    st[4] = bsize;
}
"""

_READER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t PKT = get_compile_time_arg_val(0), N = get_compile_time_arg_val(1), GROUP = get_compile_time_arg_val(2);
constexpr auto s_args = TensorAccessorArgs<3>();
void kernel_main() {
    const uint32_t sbuf = get_arg_val<uint32_t>(0), page0 = get_arg_val<uint32_t>(1);
    const auto s = TensorAccessor(s_args, sbuf, PKT);
    for (uint32_t j = 0; j < N; j += GROUP) {
        const uint32_t g = N - j < GROUP ? N - j : GROUP;
        cb_reserve_back(0, g);
        uint32_t w = get_write_ptr(0);
        for (uint32_t i = 0; i < g; ++i, w += PKT) noc_async_read(s.get_noc_addr(page0 + j + i), w, PKT);
        noc_async_read_barrier();
        cb_push_back(0, g);
    }
}
"""

_RECEIVER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    const uint32_t ctr = get_arg_val<uint32_t>(0), expect = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctr);
    noc_semaphore_wait_min(c, expect);
    noc_semaphore_inc(get_noc_addr(ctr), 0u - expect);
    noc_async_atomic_barrier();
}
"""


@pytest.mark.parametrize("mesh_device, device_params", [((2, 4), _PARAMS)], indirect=True)
def test_fabric_link_rate(mesh_device, device_params):
    payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
    assert PKT <= payload and PKT % 64 == 0, (PKT, payload)
    chips, _ = plan(mesh_device, cluster_axis=1, topology=ttnn.Topology.Linear, num_links=LINKS)
    row = {coord: ch for coord, ch in chips.items() if coord[0] == 0}
    port_xy = {(c.x, c.y) for ch in chips.values() for c in ch["ports"].values()}
    free = [xy for xy in sorted(allowed_cores(mesh_device)) if xy not in port_xy]
    recv_core = {
        (d, l): ttnn.CoreCoord(*free[i * LINKS + l]) for i, d in enumerate(("fwd", "bwd")) for l in range(LINKS)
    }
    ports = sorted(port_xy | {(c.x, c.y) for c in recv_core.values()})
    union = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in ports])
    words = SLOTS * PKT // 4 + 64  # (receive ring | results + source slot at the end)
    l1 = ttnn.from_torch(
        torch.zeros((len(ports), words), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(union, (1, words), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    src_slot = ttnn.from_torch(
        torch.zeros((len(ports), PKT // 4), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(union, (1, PKT // 4), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    base = int(l1.buffer_address())
    rbuf, res = base, base + SLOTS * PKT
    ctr_sem = ttnn.create_global_semaphore(mesh_device, union, 0)
    ctr = int(ttnn.get_global_semaphore_address(ctr_sem))
    n_streams = 4 * LINKS  # (stream id: (direction, link) -> its DRAM pages)
    page_dt = ttnn.uint32
    dram = ttnn.allocate_tensor_on_device(
        ttnn.Shape([n_streams * N, PKT // 4]), page_dt, ttnn.ROW_MAJOR_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    d_ct = list(ttnn.TensorAccessorArgs(dram).get_compile_time_args())
    dbuf = int(dram.buffer_address())
    virt = lambda c: mesh_device.worker_core_from_logical_core(c)  # noqa: E731
    node = lambda c: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*c))  # noqa: E731
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(  # noqa: E731
        processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
    )
    srcT = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    cs = lambda cores: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])  # noqa: E731
    nh = max(1, min(8, DEPTH)) if SRC == "dram" else 8
    md = ttnn.MeshProgramDescriptor()
    senders = {}  # (coord, d, l) -> logical core
    expect_in = {}  # (coord, core) -> counter increments to wait for
    sid = 0
    for coord, ch in chips.items():
        prog = ttnn.ProgramDescriptor()
        rg = ch["rings"][0]
        s_rt, r_rt, rd_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        s_cores, rd_cores = [], []
        if coord[0] == 0:
            for (j, d, l), core in ch["ports"].items():
                if DIRS == "fwd" and d != "fwd":
                    continue
                peer = rg["next"] if d == "fwd" else rg["prev"]
                if peer is None or not rg.get(f"{d}_links"):
                    continue
                rcore = recv_core[(d, l)]
                rv = virt(rcore)
                pn = node(peer)
                page0 = sid * N
                sid += 1
                args = [res, rv.x, rv.y, rbuf, ctr, dbuf, page0, int(pn.mesh_id), int(pn.chip_id)]
                args += [int(src_slot.buffer_address())]
                args += list(ttnn.setup_fabric_connection(node(coord), pn, rg[f"{d}_links"][l], prog, core))
                s_rt[core.x][core.y] = args
                s_cores.append(core)
                senders[(coord, d, l)] = core
                expect_in[(peer, (rcore.x, rcore.y))] = expect_in.get((peer, (rcore.x, rcore.y)), 0) + -(-N // INC)
                if SRC == "dram":
                    rd_rt[core.x][core.y] = [dbuf, page0]
                    rd_cores.append(core)
        kernels = []
        if s_cores:
            ct = [PKT, N, INC, int(SRC == "dram"), int(DST == "dram"), SLOTS, nh] + d_ct
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=_SENDER,
                    source_type=srcT,
                    core_ranges=cs(s_cores),
                    compile_time_args=ct,
                    runtime_args=s_rt,
                    config=dm("RISCV_0", NOC1),
                )
            )
        if rd_cores:
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=_READER,
                    source_type=srcT,
                    core_ranges=cs(rd_cores),
                    compile_time_args=[PKT, N, GROUP] + d_ct,
                    runtime_args=rd_rt,
                    config=dm("RISCV_1", NOC0),
                )
            )
            prog.cbs = [
                ttnn.CBDescriptor(
                    total_size=DEPTH * PKT,
                    core_ranges=cs(rd_cores),
                    format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.bfloat16, PKT)],
                )
            ]
        prog.kernels = kernels
        md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = prog
    # receivers (second pass: every sender is known)
    for coord in chips:
        recv = [(xy, e) for (pc, xy), e in expect_in.items() if pc == coord]
        if not recv:
            continue
        prog = md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))]
        rr = ttnn.RuntimeArgs()
        cores = []
        for xy, e in recv:
            c = ttnn.CoreCoord(*xy)
            rr[c.x][c.y] = [ctr, e]
            cores.append(c)
        ks = list(prog.kernels)
        ks.append(
            ttnn.KernelDescriptor(
                kernel_source=_RECEIVER,
                source_type=srcT,
                core_ranges=cs(cores),
                compile_time_args=[],
                runtime_args=rr,
                config=dm("RISCV_0", NOC0),
            )
        )
        prog.kernels = ks
    slots_seen = set()
    rates = {k: [] for k in senders}
    stalls = {k: [] for k in senders}
    for it in range(4):
        ttnn.generic_op([l1, dram], md)
        ttnn.synchronize_device(mesh_device)
        if it == 0:
            continue
        per = [ttnn.to_torch(t).reshape(len(ports), words) for t in ttnn.get_device_tensors(l1)]
        idx = {xy: i for i, xy in enumerate(ports)}
        for (coord, d, l), core in senders.items():
            w = per[coord[0] * 4 + coord[1]][idx[(core.x, core.y)], SLOTS * PKT // 4 :].tolist()
            t0, t1, st, nsl, bsz = (v & 0xFFFFFFFF for v in w[:5])
            slots_seen.add((nsl, bsz))
            cyc = (t1 - t0) & 0xFFFFFFFF
            rates[(coord, d, l)].append(N * PKT / (cyc / AICLK_MHZ * 1e-6) / 1e9)
            stalls[(coord, d, l)].append(st / max(cyc, 1))
    lines = []
    for k in sorted(senders):
        lines.append(
            f"{k[0]} {k[1]} link {k[2]}: {statistics.median(rates[k]):.1f} GB/s (slot wait {100 * statistics.median(stalls[k]):.0f}%)"
        )
    allr = [statistics.median(v) for v in rates.values()]
    logger.info(
        f"FLR fabric {FABRIC} payload {payload} pkt {PKT} N {N} links {LINKS} dirs {DIRS} src {SRC}"
        f"{f' group {GROUP} depth {DEPTH}' if SRC == 'dram' else ''} dst {DST}, worker channel (slots, buffer B) "
        f"{sorted(slots_seen)}: per sender min / median / max "
        f"{min(allr):.1f} / {statistics.median(allr):.1f} / {max(allr):.1f} GB/s\n  " + "\n  ".join(lines)
    )
