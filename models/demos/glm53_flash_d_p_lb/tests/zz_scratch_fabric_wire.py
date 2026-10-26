# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: the 2D fabric's on-wire rate with the combine's packets (FABRIC_2D, the LoudBox 2 x 4), one generic_op per
case. A sender core holds one fabric connection (a router on its chip, by link index) and sends N packets of B bytes
from L1 to a ring of slots on a destination core, headers routed once outside the loop (as the combine's sender),
non-blocking sends, a flush every NH sends; then one atomic increment on the destination's done word. The destination
core waits for its senders' increments. Time: the program's device time (real-time profiler), the slowest chip.

Cases (GLM_FW_CASES, default all): one link one way, two links one way, both ways, a two-hop flow forwarded by the
middle chip sharing the middle's link with the middle's own flow, and the Y (SP) axis; packet sizes GLM_FW_SIZES
(default 2048,4096,8224,8704: 8224 = the combine's 8192 B row + its 32 B trailer)."""

import os
import statistics
import time

import torch
from loguru import logger

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S_ = spec()
pytestmark = device_timeout(S_)

_SEND = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
using namespace tt::tt_fabric;
constexpr uint32_t NH = 8;
void kernel_main() {
    size_t a = 0;
    const uint32_t dst_chip = get_arg_val<uint32_t>(a++), mesh = get_arg_val<uint32_t>(a++);
    const uint32_t dx = get_arg_val<uint32_t>(a++), dy = get_arg_val<uint32_t>(a++), base = get_arg_val<uint32_t>(a++);
    const uint32_t slots = get_arg_val<uint32_t>(a++), n = get_arg_val<uint32_t>(a++), pkt = get_arg_val<uint32_t>(a++);
    const uint32_t done_addr = get_arg_val<uint32_t>(a++), src = get_arg_val<uint32_t>(a++);
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr[NH];
    for (uint32_t i = 0; i < NH; ++i) {
        hdr[i] = PacketHeaderPool::allocate_header();
        (void)fabric_set_unicast_route(hdr[i], static_cast<uint16_t>(dst_chip), static_cast<uint16_t>(mesh));
    }
    conn.open();
    for (uint32_t i = 0; i < n; ++i) {
        auto* h = hdr[i % NH];
        h->to_noc_unicast_write(NocUnicastCommandHeader{get_noc_addr(dx, dy, base + (i % slots) * pkt)}, pkt);
        conn.wait_for_empty_write_slot();
        conn.send_current_slot_non_blocking(src, pkt, reinterpret_cast<uint32_t>(h));
        if (i % NH == NH - 1) noc_async_writes_flushed();
    }
    noc_async_writes_flushed();
    hdr[0]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(dx, dy, done_addr), 1, true});
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
    const uint32_t done_addr = get_arg_val<uint32_t>(0), want = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_addr);
    while (true) {
        invalidate_l1_cache();
        if (d[0] >= want) break;
    }
    d[0] = 0;  // (the next call starts from zero)
}
"""


def _cases(C):
    # (name, [(src (r, c), dst (r, c), link, sender core (x, y))]); dst core is the same logical core as the sender's
    e = lambda c: (0, c)  # noqa: E731
    return {
        "x_1link": [(e(0), e(1), 0, (1, 1))],
        "x_2links": [(e(0), e(1), 0, (1, 1)), (e(0), e(1), 1, (2, 1))],
        "x_2links_both_ways": [
            (e(0), e(1), 0, (1, 1)),
            (e(0), e(1), 1, (2, 1)),
            (e(1), e(0), 0, (3, 1)),
            (e(1), e(0), 1, (4, 1)),
        ],
        # (0,0) -> (0,2) is forwarded by (0,1)'s router onto the (0,1) -> (0,2) link, which (0,1) also feeds itself
        "x_fwd_plus_own": [(e(0), e(2), 0, (1, 1)), (e(1), e(2), 0, (2, 1))],
        "x_2hop_alone": [(e(0), e(2), 0, (1, 1))],
        "y_1link": [((0, 0), (1, 0), 0, (1, 1))],
        "y_2links": [((0, 0), (1, 0), 0, (1, 1)), ((0, 0), (1, 0), 1, (2, 1))],
    }


@mesh_parametrize
def test_fabric_wire(mesh_device):
    import ttnn

    R, C = tuple(mesh_device.shape)
    node = lambda rc: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*rc))  # noqa: E731
    grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    SLOTS, MAXP = 12, 8704
    words = (SLOTS * MAXP + MAXP + 64) // 4
    buf = ttnn.from_torch(
        torch.zeros(1, 1, grid.x * grid.y, words, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, [1, words], ttnn.ShardOrientation.ROW_MAJOR),
        ),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    base = int(buf.buffer_address())
    src = base + SLOTS * MAXP  # (a packet's worth of source bytes)
    done = src + MAXP
    dummy = ttnn.allocate_tensor_on_device(
        ttnn.Shape([32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    recs = {}

    def cb(batch):
        for r in batch.records:
            us = (r.end_timestamp - r.start_timestamp) / (r.frequency * 1e3)
            recs[r.runtime_id] = max(recs.get(r.runtime_id, 0.0), us)

    hdl = (
        ttnn.device.RegisterProgramRealtimeProfilerCallback(cb)
        if ttnn.device.IsProgramRealtimeProfilerActive()
        else None
    )
    st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    sizes = [int(v) for v in os.environ.get("GLM_FW_SIZES", "2048,4096,8224,8704").split(",")]
    allc = _cases(C)
    names = os.environ.get("GLM_FW_CASES", ",".join(allc)).split(",")
    NPK_BYTES = 24 << 20  # (~24 MB a sender)
    try:
        for name in names:
            flows = allc[name]
            for pkt in sizes:
                n = NPK_BYTES // pkt
                progs = {}
                want = {}
                for s, d, link, core in flows:
                    want[(d, core)] = want.get((d, core), 0) + 1
                for s, d, link, core in flows:
                    p = progs.setdefault(s, ttnn.ProgramDescriptor())
                    cc = ttnn.CoreCoord(*core)
                    v = mesh_device.worker_core_from_logical_core(cc)
                    rt = ttnn.RuntimeArgs()
                    dn = node(d)
                    args = [int(dn.chip_id), int(dn.mesh_id), v.x, v.y, base, SLOTS, n, pkt, done, src]
                    # (the first hop's router: toward the destination along the dimension order (Y then X))
                    hop = (d[0], s[1]) if d[0] != s[0] else (s[0], s[1] + (1 if d[1] > s[1] else -1))
                    if d[0] != s[0]:
                        hop = (s[0] + (1 if d[0] > s[0] else -1), s[1])
                    args += list(ttnn.setup_fabric_connection(node(s), node(hop), link, p, cc))
                    rt[cc.x][cc.y] = args
                    ks = list(p.kernels)
                    ks.append(
                        ttnn.KernelDescriptor(
                            kernel_source=_SEND,
                            source_type=st,
                            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(cc, cc)]),
                            compile_time_args=[],
                            runtime_args=rt,
                            config=ttnn.DataMovementConfigDescriptor(
                                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                            ),
                        )
                    )
                    p.kernels = ks
                for (d, core), w in want.items():
                    p = progs.setdefault(d, ttnn.ProgramDescriptor())
                    cc = ttnn.CoreCoord(*core)
                    rt = ttnn.RuntimeArgs()
                    rt[cc.x][cc.y] = [done, w]
                    ks = list(p.kernels)
                    ks.append(
                        ttnn.KernelDescriptor(
                            kernel_source=_RECV,
                            source_type=st,
                            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(cc, cc)]),
                            compile_time_args=[],
                            runtime_args=rt,
                            config=ttnn.DataMovementConfigDescriptor(
                                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
                            ),
                        )
                    )
                    p.kernels = ks
                md = ttnn.MeshProgramDescriptor()
                for rc, p in progs.items():
                    md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*rc), ttnn.MeshCoordinate(*rc))] = p
                ttnn.generic_op([buf, dummy], md)  # (compile, warm)
                ttnn.synchronize_device(mesh_device)
                ids = []
                for _ in range(3):
                    i0 = ttnn._ttnn.get_device_operation_id()
                    ttnn.generic_op([buf, dummy], md)
                    ttnn.synchronize_device(mesh_device)
                    ids.append((i0, ttnn._ttnn.get_device_operation_id()))
                time.sleep(0.5)
                per = [max((recs[i] for i in range(a, b) if i in recs), default=float("nan")) for a, b in ids]
                us = statistics.median(per)
                gbs = n * pkt / (us * 1e3)
                logger.info(
                    f"FW {name:20s} pkt {pkt:5d}: {us:7.1f} us, {gbs:5.1f} GB/s a flow ({len(flows)} flows), "
                    f"{us * 1e3 / n:6.1f} ns a packet"
                )
    finally:
        if hdl is not None:
            ttnn.device.UnregisterProgramRealtimeProfilerCallback(hdl)
