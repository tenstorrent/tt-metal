# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: zz_scratch_fabric_wire's on-wire rate (its sender loop: a header a packet, wait_for_empty_write_slot, a
flush every 8) on a LoudBox 2 x 2 square as a 1 x 4 TP ring (p150_x4_ring_1x4, FABRIC_2D_TORUS_X; the square's
TT_VISIBLE_DEVICES): every edge and the wrap, each way, 1 / 2 links (GLM_FW_CASES ring_*)."""

import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn

# (zz_scratch_fabric_wire's kernels and cases, copied: that module needs a bring-up spec)
_SEND = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
using namespace tt::tt_fabric;
#ifndef FUSE_EVERY
#define FUSE_EVERY 0
#endif
#ifndef ATOM_EVERY
#define ATOM_EVERY 0
#endif
constexpr uint32_t NH = ATOM_EVERY ? 6 : 8;  // (the header pool's budget: 12 a RISC, 16 hangs)
void kernel_main() {
    size_t a = 0;
    const uint32_t dst_chip = get_arg_val<uint32_t>(a++), mesh = get_arg_val<uint32_t>(a++);
    const uint32_t dx = get_arg_val<uint32_t>(a++), dy = get_arg_val<uint32_t>(a++), base = get_arg_val<uint32_t>(a++);
    const uint32_t slots = get_arg_val<uint32_t>(a++), n = get_arg_val<uint32_t>(a++), pkt = get_arg_val<uint32_t>(a++);
    const uint32_t done_addr = get_arg_val<uint32_t>(a++), src = get_arg_val<uint32_t>(a++);
    const uint32_t nd = get_arg_val<uint32_t>(a++), d0 = a;  // (scatter: packet i to core i % nd - x | y << 8)
    a += 55;  // (always 55 words: the host pads)
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr[NH];
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hat[NH];
    for (uint32_t i = 0; i < NH; ++i) {
        hdr[i] = PacketHeaderPool::allocate_header();
        (void)fabric_set_unicast_route(hdr[i], static_cast<uint16_t>(dst_chip), static_cast<uint16_t>(mesh));
        if (ATOM_EVERY) {
            hat[i] = PacketHeaderPool::allocate_header();
            (void)fabric_set_unicast_route(hat[i], static_cast<uint16_t>(dst_chip), static_cast<uint16_t>(mesh));
        }
    }
    conn.open();
    for (uint32_t i = 0; i < n; ++i) {
        auto* h = hdr[i % NH];
        uint32_t tx = dx, ty = dy;
        if (nd) {
            const uint32_t w = get_arg_val<uint32_t>(d0 + i % nd);
            tx = w & 0xFF, ty = w >> 8;
        }
        if (FUSE_EVERY && i % (FUSE_EVERY + !FUSE_EVERY) == FUSE_EVERY - 1u) {  // (the tail's: a share's last packet + its landed count)
            h->to_noc_fused_unicast_write_atomic_inc(
                NocUnicastAtomicIncFusedCommandHeader{get_noc_addr(tx, ty, base + (i % slots) * pkt), get_noc_addr(tx, ty, done_addr + 16), 1, true}, pkt);
        } else {
            h->to_noc_unicast_write(NocUnicastCommandHeader{get_noc_addr(tx, ty, base + (i % slots) * pkt)}, pkt);
        }
        conn.wait_for_empty_write_slot();
        conn.send_current_slot_non_blocking(src, pkt, reinterpret_cast<uint32_t>(h));
        if (ATOM_EVERY && i % (ATOM_EVERY + !ATOM_EVERY) == ATOM_EVERY - 1u) {  // (the landed count as its own header-only packet)
            auto* ha = hat[i % NH];  // (its own pool: reused only after the every-NH flush)
            ha->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(tx, ty, done_addr + 16), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_non_blocking_from_address(reinterpret_cast<uint32_t>(ha), sizeof(PACKET_HEADER_TYPE));
        }
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
        # (a 1 x 4 ring - a LoudBox 2 x 2 square: p150_x4_ring_1x4 - every edge, the wrap too, each way, 1 / 2 links)
        **{
            f"ring_e{c}_{'fwd' if w else 'bwd'}_{nl}l": [
                ((0, c), (0, (c + 1) % C), l, (1 + l, 1)) if w else ((0, (c + 1) % C), (0, c), l, (1 + l, 1))
                for l in range(nl)
            ]
            for c in range(C)
            for w in (True, False)
            for nl in (1, 2)
        },
        "ring_e0_fwd_2l_scatter": [((0, 0), (0, 1), l, (1 + l, 1)) for l in range(2)],
        "ring_e0_both_2l_scatter": [((0, 0), (0, 1), l, (1 + l, 1)) for l in range(2)]
        + [((0, 1), (0, 0), l, (3 + l, 1)) for l in range(2)],
        "ring_all_2l_both": [
            f
            for c in range(C)
            for f in (
                ((0, c), (0, (c + 1) % C), 0, (1, 1 + c)),
                ((0, c), (0, (c + 1) % C), 1, (2, 1 + c)),
                ((0, (c + 1) % C), (0, c), 0, (3, 1 + c)),
                ((0, (c + 1) % C), (0, c), 1, (4, 1 + c)),
            )
        ],
    }


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        (
            (1, 4),
            dict(
                fabric_config=ttnn.FabricConfig.FABRIC_2D_TORUS_X,
                reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
            ),
        )
    ],
    indirect=True,
)
def test_ring_wire(mesh_device, device_params):
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
                    if name.endswith("_scatter"):  # (the tail's landing: ~55 engine cores, round robin)
                        sc = [
                            mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
                            for y in range(10)
                            for x in range(11)
                        ][::2][:55]
                        args += [len(sc)] + [c_.x | c_.y << 8 for c_ in sc]
                    else:
                        args += [0] + [0] * 55  # (fixed length: generic_op's cache keys ignore runtime-arg counts)
                    # (the first hop's router: toward the destination along the dimension order (Y then X))
                    hop = (d[0], s[1]) if d[0] != s[0] else (s[0], s[1] + (1 if d[1] > s[1] else -1))
                    if d[0] != s[0]:
                        hop = (s[0] + (1 if d[0] > s[0] else -1), s[1])
                    if abs(d[1] - s[1]) in (1, C - 1) and d[0] == s[0]:  # (a neighbour, the ring's wrap too)
                        hop = d
                    args += list(ttnn.setup_fabric_connection(node(s), node(hop), link, p, cc))
                    rt[cc.x][cc.y] = args
                    ks = list(p.kernels)
                    ks.append(
                        ttnn.KernelDescriptor(
                            kernel_source=f"#define FUSE_EVERY {int(os.environ.get('FW_FUSE_EVERY', '0'))}\n"
                            f"#define ATOM_EVERY {int(os.environ.get('FW_ATOM_EVERY', '0'))}\n" + _SEND,
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
