# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""tp_gather - the MoE head's x all-gather over the TP line, by width (dim 3), store-and-forward over fabric unicasts.

Every chip p of a mesh row (positions 0 .. G-1 along cluster axis 1) holds x [Rw, kl] (TILE, DRAM interleaved) and
gets x_full [Rw, G kl] with chip q's columns at q kl. Per direction (east: toward p+1, west: toward p-1) and link, one
port core sends one hop: first its own columns (read from x), then, in order, what the port upstream in the same
direction landed here (read back from x_full once its arrival counter covers it) - east: p, p-1, .., 0; west: p, p+1,
.., G-1. A packet is a run of tiles that sit consecutively in one DRAM bank in x AND in x_full (row tile r, bank b:
tiles c = b, b + B, ..; kl / 32 a multiple of the bank count B), written at the same address on the neighbour. Every
INC_EVERY-th packet of a stream (and its last) is a fused write + increment of the downstream port's counter. A copy
core per link writes the chip's own columns. The busiest link carries the same bytes as a line multicast would
(measured: ~15.6 GB/s a link a direction at 200G, so 2 links on the 4-chip line bound GLM-5.3-Flash's x at ~335 us).

Fence (a call's): no port writes into a neighbour's x_full before that neighbour's port facing it started the same
call. Ports and copy cores: combine_rf.fabric_all_gather_py.plan (a port per direction and link, placed by its
Ethernet core; line ends keep a receive-only port)."""

from ttnn.bringup.combine_rf.fabric_all_gather_py import _OPP, NOC0, NOC1, plan

import ttnn

INC_EVERY = 8
CB = 0

_WALK = r"""
// one shard's runs for this port, in order: row tile r, bank b (b0, b0 + BS, ..), up to RP pages a packet
template <typename F>
FORCE_INLINE void walk(uint32_t rows, uint32_t b0, uint32_t bs, F&& f) {
    constexpr uint32_t PB = CT / NB;
    uint32_t idx = 0;
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t b = b0; b < NB; b += bs)
            for (uint32_t i = 0; i < PB; i += RP) f(r * CT + b + i * NB, PB - i < RP ? PB - i : RP, idx++);
}
// a shard-local page (row tile r, column c) -> its page in x_full, shard q
FORCE_INLINE uint32_t out_page(uint32_t page, uint32_t q) { return (page / CT) * G * CT + q * CT + page % CT; }
"""

_READER = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t CT = get_compile_time_arg_val(0), NB = get_compile_time_arg_val(1), RP = get_compile_time_arg_val(2);
constexpr uint32_t GROUP = get_compile_time_arg_val(3), INC = get_compile_time_arg_val(4), G = get_compile_time_arg_val(5);
constexpr auto x_args = TensorAccessorArgs<6>();
constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
"""
    + _WALK
    + r"""
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0), o_addr = get_arg_val<uint32_t>(1), rows = get_arg_val<uint32_t>(2);
    const uint32_t b0 = get_arg_val<uint32_t>(3), bs = get_arg_val<uint32_t>(4), arrival = get_arg_val<uint32_t>(5);
    const uint32_t full = get_arg_val<uint32_t>(6), n = get_arg_val<uint32_t>(7);  // packets a shard; shards to send
    const auto x = TensorAccessor(x_args, x_addr, 2048);
    const auto o = TensorAccessor(o_args, o_addr, 2048);
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival);
    const uint32_t total = n * full;
    uint32_t batch = 0, wptr = 0, done = 0;
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t q = get_arg_val<uint32_t>(8 + k);
        walk(rows, b0, bs, [&](uint32_t page, uint32_t np, uint32_t idx) {
            if (batch == 0) {
                cb_reserve_back(0, RP * GROUP);
                wptr = get_write_ptr(0);
            }
            uint64_t src;
            if (k == 0) {
                src = x.get_noc_addr(page);  // my own columns
            } else {  // relay: upstream's shard k - 1 of its stream landed here (its packet (k - 1) full + idx)
                noc_semaphore_wait_min(arrived, ((k - 1) * full + idx) / INC + 1);
                src = o.get_noc_addr(out_page(page, q));
            }
            noc_async_read(src, wptr + batch * RP * 2048, np * 2048);
            ++done;
            if (++batch == GROUP || done == total) {
                noc_async_read_barrier();
                cb_push_back(0, RP * GROUP);
                batch = 0;
            }
        });
    }
}
"""
)

_SENDER = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
constexpr uint32_t CT = get_compile_time_arg_val(0), NB = get_compile_time_arg_val(1), RP = get_compile_time_arg_val(2);
constexpr uint32_t GROUP = get_compile_time_arg_val(3), INC = get_compile_time_arg_val(4), G = get_compile_time_arg_val(5);
constexpr auto o_args = TensorAccessorArgs<6>();
"""
    + _WALK
    + r"""
void kernel_main() {
    size_t a = 0;
    const uint32_t o_addr = get_arg_val<uint32_t>(a++), rows = get_arg_val<uint32_t>(a++);
    const uint32_t b0 = get_arg_val<uint32_t>(a++), bs = get_arg_val<uint32_t>(a++);
    const uint32_t arrival = get_arg_val<uint32_t>(a++), expect_in = get_arg_val<uint32_t>(a++);  // (my counter)
    const uint32_t ready = get_arg_val<uint32_t>(a++), wait_ready = get_arg_val<uint32_t>(a++);
    const uint32_t send_ready = get_arg_val<uint32_t>(a++), rx = get_arg_val<uint32_t>(a++), ry = get_arg_val<uint32_t>(a++);
    const uint32_t px = get_arg_val<uint32_t>(a++), py = get_arg_val<uint32_t>(a++);  // downstream port (NoC)
    const uint16_t mesh = get_arg_val<uint32_t>(a++), chip = get_arg_val<uint32_t>(a++);  // the neighbour
    const uint32_t full = get_arg_val<uint32_t>(a++), n = get_arg_val<uint32_t>(a++);
    const uint32_t qs = a;
    a += n;
    const auto o = TensorAccessor(o_args, o_addr, 2048);
    if (n > 0 || send_ready) {
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[GROUP];
        for (uint32_t h = 0; h < GROUP; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
            (void)fabric_set_unicast_route(hdrs[h], chip, mesh);
        }
        conn.open();
        // fence: I started (whatever read my x_full last call is done): tell the neighbour's port that sends to me
        if (send_ready) {
            hdrs[0]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(rx, ry, ready), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdrs[0]), sizeof(PACKET_HEADER_TYPE));
        }
        if (wait_ready) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready), 1);
            noc_semaphore_inc(get_noc_addr(ready), 0u - 1u);
            noc_async_atomic_barrier();
        }
        const uint64_t arr_noc = get_noc_addr(px, py, arrival);
        const uint32_t total = n * full;
        uint32_t sent = 0, h = 0, unflushed = 0;
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t q = get_arg_val<uint32_t>(qs + k);
            walk(rows, b0, bs, [&](uint32_t page, uint32_t np, uint32_t) {
                cb_wait_front(0, RP * (unflushed + 1));
                const uint32_t src = get_read_ptr(0) + unflushed * RP * 2048;
                const uint64_t dst = o.get_noc_addr(out_page(page, q), 0, 0);
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                h = h + 1 == GROUP ? 0 : h + 1;
                if ((++sent % INC) == 0 || sent == total) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(NocUnicastAtomicIncFusedCommandHeader{dst, arr_noc, 1, true}, np * 2048);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, np * 2048);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(src, np * 2048, reinterpret_cast<uint32_t>(hdr));
                if (++unflushed == GROUP || sent == total) {
                    noc_async_writes_flushed();
                    cb_pop_front(0, RP * GROUP);
                    unflushed = 0;
                }
            });
        }
        conn.close();
    }
    // everything from upstream has landed (the relays needed part of it already); re-arm for the next call
    if (expect_in > 0) {
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival);
        noc_semaphore_wait_min(arrived, expect_in);
        noc_semaphore_inc(get_noc_addr(arrival), 0u - expect_in);
        noc_async_atomic_barrier();
    }
}
"""
)

_COPY = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t CT = get_compile_time_arg_val(0), NB = get_compile_time_arg_val(1), RP = get_compile_time_arg_val(2);
constexpr uint32_t G = get_compile_time_arg_val(3);
constexpr auto x_args = TensorAccessorArgs<4>();
constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
"""
    + _WALK
    + r"""
// my own columns into x_full: the runs of banks b0, b0 + BS, .. read from x, written to the same bank in x_full
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0), o_addr = get_arg_val<uint32_t>(1), rows = get_arg_val<uint32_t>(2);
    const uint32_t b0 = get_arg_val<uint32_t>(3), bs = get_arg_val<uint32_t>(4), p = get_arg_val<uint32_t>(5);
    const auto x = TensorAccessor(x_args, x_addr, 2048);
    const auto o = TensorAccessor(o_args, o_addr, 2048);
    const uint32_t buf = get_write_ptr(0);
    uint32_t slot = 0;
    walk(rows, b0, bs, [&](uint32_t page, uint32_t np, uint32_t) {
        const uint32_t l1 = buf + slot * RP * 2048;
        noc_async_read(x.get_noc_addr(page), l1, np * 2048);
        noc_async_read_barrier();
        noc_async_write(l1, o.get_noc_addr(out_page(page, p)), np * 2048);
        slot = (slot + 1) % 4;
        if (slot == 0) noc_async_writes_flushed();
    });
    noc_async_write_barrier();
}
"""
)


def _incs(n):
    return n // INC_EVERY + (1 if n % INC_EVERY else 0)


def tp_gather_prepare(mesh_device, rows, kl, *, num_links=2, group=4):
    """Returns run(x [1, 1, rows, kl] bf16 TILE DRAM) -> x_full [1, 1, rows, G kl] (a fresh DRAM tensor)."""
    G = tuple(mesh_device.shape)[1]
    nb = mesh_device.dram_grid_size().x
    ct, rt = kl // 32, rows // 32
    assert kl % 32 == 0 and rows % 32 == 0 and ct % nb == 0, f"kl / 32 = {ct} must be a multiple of the {nb} banks"
    rp = max(1, min(ct // nb, ttnn.get_tt_fabric_max_payload_size_bytes() // 2048))
    assert 4 * rp <= 2 * group * rp, "the copy core's 4 slots must fit the CB"
    chips, _ = plan(mesh_device, cluster_axis=1, topology=ttnn.Topology.Linear, num_links=num_links)
    ports_all = sorted({(c.x, c.y) for ch in chips.values() for c in ch["ports"].values()})
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in ports_all])
    sems = [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(2)]  # arrival, ready (on every port core)
    arrival, ready = (int(ttnn.get_global_semaphore_address(s)) for s in sems)
    per_shard = lambda l: rt * len(range(l, nb, num_links)) * -(-(ct // nb) // rp)  # noqa: E731
    virt = lambda c: mesh_device.worker_core_from_logical_core(c)  # noqa: E731
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))  # noqa: E731
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(  # noqa: E731
        processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
    )
    src = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    state = dict(key=None)

    def sends(p, d):  # the shards a port sends, own first (east: p, p-1, .., 0; west: p, p+1, .., G-1)
        if d == "fwd":
            return list(range(p, -1, -1)) if p < G - 1 else []
        return list(range(p, G)) if p > 0 else []

    def build(x, out):
        x_ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
        o_ct = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
        xa, oa = int(x.buffer_address()), int(out.buffer_address())
        md = ttnn.MeshProgramDescriptor()
        for coord, ch in chips.items():
            rg = ch["rings"][0]
            p = rg["p"]
            prog = ttnn.ProgramDescriptor()
            port_cores = list(ch["ports"].values())
            prog.cbs = [
                ttnn.CBDescriptor(
                    total_size=2 * group * rp * 2048,
                    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in port_cores + ch["copy"]]),
                    format_descriptors=[ttnn.CBFormatDescriptor(CB, ttnn.bfloat16, 2048)],
                )
            ]
            rd_rt, sd_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for (_, d, l), core in ch["ports"].items():
                peer = rg["next"] if d == "fwd" else rg["prev"]
                up = rg["prev"] if d == "fwd" else rg["next"]
                mine = sends(p, d) if peer is not None else []
                up_n = len(sends(chips[up]["rings"][0]["p"], d)) if up is not None else 0
                full = per_shard(l)
                rd_rt[core.x][core.y] = [xa, oa, rt, l, num_links, arrival, full, len(mine)] + mine
                args = [oa, rt, l, num_links, arrival, _incs(up_n * full) if up_n else 0, ready, int(bool(mine))]
                # (my ready goes to the neighbour's port facing me: it sends to me)
                if peer is not None and rg.get(f"{d}_links"):
                    rc = virt(chips[peer]["ports"][(0, _OPP[d], l)])
                    pc = virt(chips[peer]["ports"][(0, d, l)])
                    pn = node(peer)
                    send_ready = bool(sends(chips[peer]["rings"][0]["p"], _OPP[d]))
                    args += [int(send_ready), rc.x, rc.y, pc.x, pc.y, int(pn.mesh_id), int(pn.chip_id), full, len(mine)]
                    args += mine
                    args += list(ttnn.setup_fabric_connection(node(coord), pn, rg[f"{d}_links"][l], prog, core))
                else:
                    args += [0, 0, 0, 0, 0, 0, 0, full, 0]
                sd_rt[core.x][core.y] = args
            cp_rt = ttnn.RuntimeArgs()
            for j, core in enumerate(ch["copy"]):
                cp_rt[core.x][core.y] = [xa, oa, rt, j, len(ch["copy"]), p]
            cs = lambda cores: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])  # noqa: E731
            ctk = [ct, nb, rp, group, INC_EVERY, G]
            prog.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=_READER,
                    source_type=src,
                    core_ranges=cs(port_cores),
                    compile_time_args=ctk + x_ct + o_ct,
                    runtime_args=rd_rt,
                    config=dm("RISCV_1", NOC0),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=_SENDER,
                    source_type=src,
                    core_ranges=cs(port_cores),
                    compile_time_args=ctk + o_ct,
                    runtime_args=sd_rt,
                    config=dm("RISCV_0", NOC1),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=_COPY,
                    source_type=src,
                    core_ranges=cs(ch["copy"]),
                    compile_time_args=[ct, nb, rp, G] + x_ct + o_ct,
                    runtime_args=cp_rt,
                    config=dm("RISCV_0", NOC0),
                ),
            ]
            r, c = coord
            md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = prog
        return md

    def run(x, out=None):
        if out is None:
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, rows, G * kl]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
            )
        k = (int(x.buffer_address()), int(out.buffer_address()))
        if state["key"] != k:
            state["key"], state["desc"] = k, build(x, out)
        ttnn.generic_op([x, out], state["desc"])
        return out

    run.sems = sems  # (kept alive with the op)
    return run
