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
core per link writes the chip's own columns. The busiest link carries the same bytes as a line multicast would: on a
line of G the end links carry G - 1 shards, on a balanced ring G / 2 - 1/2 (1.5 at G = 4). Links run at ~23.6 GB/s a
direction at 200G, so 2 links on the 4-chip line bound GLM-5.3-Flash's x at 3 x 5.24 MB / 47 GB/s = 333 us (336 measured).

Fence (a call's): no port writes into a neighbour's x_full before that neighbour's port facing it started the same
call. Ports and copy cores: combine_rf.fabric_all_gather_py.plan (a port per direction and link, placed by its
Ethernet core; line ends keep a receive-only port)."""

from ttnn.bringup.combine_rf.fabric_all_gather_py import _OPP, NOC0, NOC1, plan

import ttnn

INC_EVERY = 8
CB = 0

_WALK = r"""
// one shard's runs for this port in row tiles r0 .. r1 - 1 (a chunk), in order: row tile r, bank b (b0, b0 + BS, ..),
// up to RP pages a packet. part 0: every row tile; 1 / 2: the even / odd row tiles (a ring's opposite shard, half each
// way). idx: the run's index in the chunk's full walk (part 0) of the shard (the upstream sent it whole: host-checked)
template <typename F>
FORCE_INLINE void walk(uint32_t r0, uint32_t r1, uint32_t b0, uint32_t bs, uint32_t part, F&& f) {
    constexpr uint32_t PB = CT / NB;
    uint32_t idx = 0;
    for (uint32_t r = r0; r < r1; ++r)
        for (uint32_t b = b0; b < NB; b += bs)
            for (uint32_t i = 0; i < PB; i += RP, ++idx)
                if (part == 0 || (r & 1) == part - 1) f(r * CT + b + i * NB, PB - i < RP ? PB - i : RP, idx);
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
constexpr uint32_t CR = get_compile_time_arg_val(6), MC = get_compile_time_arg_val(7);  // (row tiles a chunk, chunks)
constexpr auto x_args = TensorAccessorArgs<8>();
constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
"""
    + _WALK
    + r"""
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0), o_addr = get_arg_val<uint32_t>(1);
    const uint32_t b0 = get_arg_val<uint32_t>(2), bs = get_arg_val<uint32_t>(3), arrival = get_arg_val<uint32_t>(4);
    const uint32_t total_c = get_arg_val<uint32_t>(5), n = get_arg_val<uint32_t>(6);  // packets I send a chunk; entries
    const uint32_t up_ipc = get_arg_val<uint32_t>(7);  // (the upstream stream's increments a chunk)
    const auto x = TensorAccessor(x_args, x_addr, 2048);
    const auto o = TensorAccessor(o_args, o_addr, 2048);
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival);
    uint32_t batch = 0, wptr = 0;
    for (uint32_t m = 0; m < MC; ++m) {
        uint32_t done = 0;
        for (uint32_t k = 0; k < n; ++k) {  // entry k: shard q, part, (relays) its first packet in upstream's chunk
            const uint32_t q = get_arg_val<uint32_t>(8 + 3 * k), part = get_arg_val<uint32_t>(9 + 3 * k);
            const uint32_t up_base = get_arg_val<uint32_t>(10 + 3 * k);
            walk(m * CR, (m + 1) * CR, b0, bs, part, [&](uint32_t page, uint32_t np, uint32_t idx) {
                if (batch == 0) {
                    cb_reserve_back(0, RP * GROUP);
                    wptr = get_write_ptr(0);
                }
                uint64_t src;
                if (k == 0) {
                    src = x.get_noc_addr(page);  // my own columns
                } else {  // relay: upstream's packet up_base + idx of chunk m (increments counted per chunk) landed
                    noc_semaphore_wait_min(arrived, m * up_ipc + (up_base + idx) / INC + 1);
                    src = o.get_noc_addr(out_page(page, q));
                }
                noc_async_read(src, wptr + batch * RP * 2048, np * 2048);
                ++batch;
                ++done;
                if (batch == GROUP || done == total_c) {  // (a chunk's end ends a batch)
                    noc_async_read_barrier();
                    cb_push_back(0, RP * GROUP);
                    batch = 0;
                }
            });
        }
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
constexpr uint32_t CR = get_compile_time_arg_val(6), MC = get_compile_time_arg_val(7);
constexpr auto o_args = TensorAccessorArgs<8>();
"""
    + _WALK
    + r"""
void kernel_main() {
    size_t a = 0;
    const uint32_t o_addr = get_arg_val<uint32_t>(a++);
    const uint32_t b0 = get_arg_val<uint32_t>(a++), bs = get_arg_val<uint32_t>(a++);
    const uint32_t arrival = get_arg_val<uint32_t>(a++), expect_in = get_arg_val<uint32_t>(a++);  // (my counter)
    const uint32_t ready = get_arg_val<uint32_t>(a++), wait_ready = get_arg_val<uint32_t>(a++);
    const uint32_t send_ready = get_arg_val<uint32_t>(a++), rx = get_arg_val<uint32_t>(a++), ry = get_arg_val<uint32_t>(a++);
    const uint32_t px = get_arg_val<uint32_t>(a++), py = get_arg_val<uint32_t>(a++);  // downstream port (NoC)
    const uint16_t mesh = get_arg_val<uint32_t>(a++), chip = get_arg_val<uint32_t>(a++);  // the neighbour
    const uint32_t total_c = get_arg_val<uint32_t>(a++), n = get_arg_val<uint32_t>(a++);  // (packets a chunk, entries)
    const uint32_t watch_on = get_arg_val<uint32_t>(a++), wx = get_arg_val<uint32_t>(a++), wy = get_arg_val<uint32_t>(a++);
    const uint32_t watch = get_arg_val<uint32_t>(a++);  // (the neighbour's watcher: one increment a chunk)
    const uint32_t qs = a;
    a += 2 * n;
    const auto o = TensorAccessor(o_args, o_addr, 2048);
    if (n > 0 || send_ready) {
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[GROUP + 1];
        for (uint32_t h = 0; h <= GROUP; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
            (void)fabric_set_unicast_route(hdrs[h], chip, mesh);
        }
        conn.open();
        // fence: I started (whatever read my x_full last call is done): tell the neighbour's port that sends to me
        if (send_ready) {
            hdrs[GROUP]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(rx, ry, ready), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdrs[GROUP]), sizeof(PACKET_HEADER_TYPE));
        }
        if (wait_ready) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready), 1);
            noc_semaphore_inc(get_noc_addr(ready), 0u - 1u);
            noc_async_atomic_barrier();
        }
        const uint64_t arr_noc = get_noc_addr(px, py, arrival);
        uint32_t h = 0, unflushed = 0;
        for (uint32_t m = 0; m < MC && n > 0; ++m) {
            uint32_t sent = 0;  // (this chunk's: increments every INC-th packet of a chunk and at its end)
            for (uint32_t k = 0; k < n; ++k) {
                const uint32_t q = get_arg_val<uint32_t>(qs + 2 * k), part = get_arg_val<uint32_t>(qs + 2 * k + 1);
                walk(m * CR, (m + 1) * CR, b0, bs, part, [&](uint32_t page, uint32_t np, uint32_t) {
                    cb_wait_front(0, RP * (unflushed + 1));
                    const uint32_t src = get_read_ptr(0) + unflushed * RP * 2048;
                    const uint64_t dst = o.get_noc_addr(out_page(page, q), 0, 0);
                    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                    h = h + 1 == GROUP ? 0 : h + 1;
                    if ((++sent % INC) == 0 || sent == total_c) {
                        hdr->to_noc_fused_unicast_write_atomic_inc(
                            NocUnicastAtomicIncFusedCommandHeader{dst, arr_noc, 1, true}, np * 2048);
                    } else {
                        hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, np * 2048);
                    }
                    conn.wait_for_empty_write_slot();
                    conn.send_current_slot_non_blocking(src, np * 2048, reinterpret_cast<uint32_t>(hdr));
                    if (++unflushed == GROUP || sent == total_c) {
                        noc_async_writes_flushed();
                        cb_pop_front(0, RP * GROUP);
                        unflushed = 0;
                    }
                });
            }
            if (watch_on) {  // chunk m is all sent: the neighbour's watcher (behind the chunk's packets on this link)
                hdrs[GROUP]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(wx, wy, watch), 1, true});
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdrs[GROUP]), sizeof(PACKET_HEADER_TYPE));
            }
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
constexpr uint32_t G = get_compile_time_arg_val(3), CR = get_compile_time_arg_val(4), MC = get_compile_time_arg_val(5);
constexpr uint32_t GROUP = get_compile_time_arg_val(6);
constexpr auto x_args = TensorAccessorArgs<7>();
constexpr auto o_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
"""
    + _WALK
    + r"""
// my own columns into x_full: the runs of banks b0, b0 + BS, .. read from x, written to the same bank in x_full; GROUP
// runs in flight (two halves of the buffer: one reading while the other's writes drain); a chunk at a time, each
// chunk's end (written) counted on the watcher
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0), o_addr = get_arg_val<uint32_t>(1);
    const uint32_t b0 = get_arg_val<uint32_t>(2), bs = get_arg_val<uint32_t>(3), p = get_arg_val<uint32_t>(4);
    const uint32_t watch_on = get_arg_val<uint32_t>(5), wx = get_arg_val<uint32_t>(6), wy = get_arg_val<uint32_t>(7);
    const uint32_t watch = get_arg_val<uint32_t>(8);
    const auto x = TensorAccessor(x_args, x_addr, 2048);
    const auto o = TensorAccessor(o_args, o_addr, 2048);
    const uint32_t buf = get_write_ptr(0);
    uint32_t half = 0, nb = 0;
    uint32_t pages[GROUP], nps[GROUP];
    auto drain = [&]() {  // the half's reads landed: write them out
        noc_async_read_barrier();
        noc_async_writes_flushed();  // (the other half's writes, a batch ago: it is refilled next)
        const uint32_t base = buf + half * GROUP * RP * 2048;
        for (uint32_t i = 0; i < nb; ++i) noc_async_write(base + i * RP * 2048, o.get_noc_addr(out_page(pages[i], p)), nps[i] * 2048);
        half ^= 1;
        nb = 0;
    };
    for (uint32_t m = 0; m < MC; ++m) {
        walk(m * CR, (m + 1) * CR, b0, bs, 0, [&](uint32_t page, uint32_t np, uint32_t) {
            noc_async_read(x.get_noc_addr(page), buf + (half * GROUP + nb) * RP * 2048, np * 2048);
            pages[nb] = page;
            nps[nb] = np;
            if (++nb == GROUP) drain();
        });
        if (nb) drain();
        noc_async_write_barrier();
        if (watch_on) noc_semaphore_inc(get_noc_addr(wx, wy, watch), 1);
    }
    noc_async_atomic_barrier();
}
"""
)

_WATCHER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
// chunk m of x_full is complete on this chip once every incoming stream (its chunk-end increment, behind the chunk's
// packets) and every copy core counted it: then one increment of semaphore SEM on each consumer core
void kernel_main() {
    const uint32_t watch = get_arg_val<uint32_t>(0), per = get_arg_val<uint32_t>(1), mc = get_arg_val<uint32_t>(2);
    const uint32_t sem = get_semaphore(get_arg_val<uint32_t>(3)), n = get_arg_val<uint32_t>(4);
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(watch);
    for (uint32_t m = 0; m < mc; ++m) {
        noc_semaphore_wait_min(w, per * (m + 1));
        for (uint32_t i = 0; i < n; ++i)
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(5 + 2 * i), get_arg_val<uint32_t>(6 + 2 * i), sem), 1);
    }
    noc_semaphore_inc(get_noc_addr(watch), 0u - per * mc);  // (re-arm for the next call)
    noc_async_atomic_barrier();
}
"""


def _incs(n):
    return n // INC_EVERY + (1 if n % INC_EVERY else 0)


class TpGather:
    """The x all-gather over the TP axis (module docstring) as pieces of a per-chip program, a chunk of chunk_rows rows
    at a time (chunk-major streams: the consumer can start on chunk 0). Options:
      num_links: links a direction, one port core (stream) a link direction - FABRIC_2D allows one worker a link
        direction (2 on one link hung), and one already reaches ~95% of the link;
    Packets: whole tiles, up to the fabric's configured max payload (the mesh opens it from the model's width).
      group: packets a reader batch; ring: the TP axis wraps (None: from the fabric config);
      sub_core_grid: the cores the ports / copy cores may use (None: the whole grid); placement: plan's ("auto": ports
        by their Ethernet cores, needs a row of 4 allowed cores; "simple": any free core)."""

    def __init__(
        self,
        mesh_device,
        rows,
        kl,
        *,
        chunk_rows=None,
        num_links=2,
        group=4,
        ring=None,
        sub_core_grid=None,
        placement="auto",
    ):
        self.mesh = mesh_device
        G = self.G = tuple(mesh_device.shape)[1]
        nb = self.nb = mesh_device.dram_grid_size().x
        ct, rt = kl // 32, rows // 32
        self.rows, self.kl, self.ct, self.rt = rows, kl, ct, rt
        assert kl % 32 == 0 and rows % 32 == 0 and ct % nb == 0, f"kl / 32 = {ct} must be a multiple of the {nb} banks"
        chunk_rows = chunk_rows or rows
        assert chunk_rows % 32 == 0 and rows % chunk_rows == 0
        self.cr, self.mc = chunk_rows // 32, rows // chunk_rows
        self.rp = rp = max(1, min(ct // nb, ttnn.get_tt_fabric_max_payload_size_bytes() // 2048))
        assert (
            group + 1 <= 12
        ), "the sender takes group + 1 packet headers: Blackhole's pool has 12 a RISC (spins if over)"
        self.group = group
        nv = self.nv = num_links  # streams a direction: stream v on link v, banks v, v + nv, ..
        assert nv <= nb, "a stream needs a bank"
        if ring is None:
            cfg = ttnn.get_fabric_config()
            ring = cfg in (ttnn.FabricConfig.FABRIC_2D_TORUS_X, ttnn.FabricConfig.FABRIC_2D_TORUS_XY) and G > 2
        self.ring = ring
        topo = ttnn.Topology.Ring if ring else ttnn.Topology.Linear
        self.chips, _ = plan(
            mesh_device,
            cluster_axis=1,
            topology=topo,
            num_links=num_links,
            balance=ring,
            sub_core_grid=sub_core_grid,
            placement=placement,
        )
        self.num_links = num_links
        self.ports = {}  # coord -> {(d, v): core}
        for coord, ch in self.chips.items():
            self.ports[coord] = {(d, l): core for (_, d, l), core in ch["ports"].items()}
            for d in ("fwd", "bwd"):  # a relay's upstream entry is whole (the walk's idx assumes it)
                sched = ch["rings"][0][d]
                assert all(part == 0 for _, part in sched[:-1]), sched
        ports_all = sorted(
            {(c.x, c.y) for pv in self.ports.values() for c in pv.values()}
            | {(c.x, c.y) for ch in self.chips.values() for c in ch["copy"]}
        )
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in ports_all])
        self.sems = [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(3)]  # arrival, ready, watch
        self.arrival, self.ready, self.watch = (int(ttnn.get_global_semaphore_address(s)) for s in self.sems)
        self.pieces_ = -(-(ct // nb) // rp)
        self.state = dict(key=None)

    def _pk(self, v, part, m):
        """Packets of an entry (stream v, part) in chunk m."""
        rows = [r for r in range(m * self.cr, (m + 1) * self.cr) if part == 0 or (r & 1) == part - 1]
        return len(rows) * len(range(v, self.nb, self.nv)) * self.pieces_

    def watcher_core(self, coord):
        return self.chips[coord]["copy"][0]

    def pieces(self, coord, prog, x, out, notify=None):
        """(CBs, kernels) of chip `coord` (fabric connections are set up on prog). notify: (logical consumer cores,
        semaphore id): one increment a chunk on each once the chunk of x_full is complete here (a watcher kernel on the
        first copy core's NCRISC; the caller declares that semaphore id on the watcher core too)."""
        mesh_device, chips, G, nb, nv = self.mesh, self.chips, self.G, self.nb, self.nv
        ct, rp, group, mc, cr = self.ct, self.rp, self.group, self.mc, self.cr
        virt = lambda c: mesh_device.worker_core_from_logical_core(c)  # noqa: E731
        node = lambda c: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*c))  # noqa: E731
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(  # noqa: E731
            processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
        )
        src = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
        x_ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
        o_ct = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
        xa, oa = int(x.buffer_address()), int(out.buffer_address())
        ch = chips[coord]
        rg = ch["rings"][0]
        pv = self.ports[coord]
        port_cores = list(pv.values())
        cs = lambda cores: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])  # noqa: E731
        cbs = [
            ttnn.CBDescriptor(
                total_size=2 * group * rp * 2048,
                core_ranges=cs(port_cores + ch["copy"]),
                format_descriptors=[ttnn.CBFormatDescriptor(CB, ttnn.bfloat16, 2048)],
            )
        ]
        watch_on = notify is not None
        rd_rt, sd_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        n_in = 0
        for (d, v), core in pv.items():
            l = v % self.num_links
            peer = rg["next"] if d == "fwd" else rg["prev"]
            up = rg["prev"] if d == "fwd" else rg["next"]
            mine = rg[d] if peer is not None else []  # [(shard, part)], own first
            ups = chips[up]["rings"][0][d] if up is not None else []
            n_in += bool(ups)
            up_cum = [0]  # (per chunk: every chunk has the same rows, hence the same packets; checked)
            for _, part in ups:
                up_cum.append(up_cum[-1] + self._pk(v, part, 0))
            assert all(sum(self._pk(v, part, m) for _, part in ups) == up_cum[-1] for m in range(mc))
            total_c = sum(self._pk(v, part, 0) for _, part in mine)
            ent = []
            for k, (q, part) in enumerate(mine):
                ent += [q, part, up_cum[k - 1] if k > 0 else 0]
            up_ipc = _incs(up_cum[-1])
            rd_rt[core.x][core.y] = [xa, oa, v, nv, self.arrival, total_c, len(mine), up_ipc] + ent
            args = [oa, v, nv, self.arrival, mc * up_ipc if ups else 0, self.ready, int(bool(mine))]
            if peer is not None and rg.get(f"{d}_links"):
                rc = virt(self.ports[peer][(_OPP[d], v)])
                pc = virt(self.ports[peer][(d, v)])
                wc = virt(self.watcher_core(peer))
                pn = node(peer)
                send_ready = bool(chips[peer]["rings"][0][_OPP[d]])
                args += [
                    int(send_ready),
                    rc.x,
                    rc.y,
                    pc.x,
                    pc.y,
                    int(pn.mesh_id),
                    int(pn.chip_id),
                    total_c,
                    len(mine),
                    int(watch_on),
                    wc.x,
                    wc.y,
                    self.watch,
                ]
                args += [val for q, part in mine for val in (q, part)]
                args += list(ttnn.setup_fabric_connection(node(coord), pn, rg[f"{d}_links"][l], prog, core))
            else:
                args += [0] * 13
            sd_rt[core.x][core.y] = args
        wv = virt(self.watcher_core(coord))
        cp_rt = ttnn.RuntimeArgs()
        for j, core in enumerate(ch["copy"]):
            cp_rt[core.x][core.y] = [xa, oa, j, len(ch["copy"]), rg["p"], int(watch_on), wv.x, wv.y, self.watch]
        ctk = [ct, nb, rp, group, INC_EVERY, G, cr, mc]
        kernels = [
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
                compile_time_args=[ct, nb, rp, G, cr, mc, group] + x_ct + o_ct,
                runtime_args=cp_rt,
                config=dm("RISCV_0", NOC0),
            ),
        ]
        if watch_on:
            cons, sem_id = notify
            wt = ttnn.RuntimeArgs()
            wc = self.watcher_core(coord)
            ta = []
            for c in cons:
                vc = virt(c)
                ta += [vc.x, vc.y]
            wt[wc.x][wc.y] = [self.watch, n_in + len(ch["copy"]), mc, sem_id, len(cons)] + ta
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=_WATCHER,
                    source_type=src,
                    core_ranges=cs([wc]),
                    compile_time_args=[],
                    runtime_args=wt,
                    config=dm("RISCV_1", NOC1),
                )
            )
        return cbs, kernels

    def alloc_out(self):
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.rows, self.G * self.kl]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.mesh,
            ttnn.DRAM_MEMORY_CONFIG,
        )

    def run(self, x, out=None):
        """The gather alone: x [1, 1, rows, kl] bf16 TILE DRAM -> x_full [1, 1, rows, G kl]."""
        out = out if out is not None else self.alloc_out()
        k = (int(x.buffer_address()), int(out.buffer_address()))
        if self.state["key"] != k:
            md = ttnn.MeshProgramDescriptor()
            for coord in self.chips:
                prog = ttnn.ProgramDescriptor()
                prog.cbs, prog.kernels = self.pieces(coord, prog, x, out)
                r, c = coord
                md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = prog
            self.state["key"], self.state["desc"] = k, md
        ttnn.generic_op([x, out], self.state["desc"])
        return out


def tp_gather_prepare(mesh_device, rows, kl, **opts):
    """run(x) -> x_full (TpGather.run); opts: TpGather's."""
    tg = TpGather(mesh_device, rows, kl, **opts)
    run = tg.run
    return run
