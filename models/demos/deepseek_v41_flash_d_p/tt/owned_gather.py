# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PACKED_HIST_OWNED=on (PB.HG, OWNER 2026-11-21): a chip-aligned packed chunk's per-segment compressed history goes
ONLY to the chips that segment owns, every chip receiving exactly one history (its own segment's) into ONE buffer a
KV source.

Today (off) every (source, segment) history is all-gathered over the whole mesh into the segment's own lane buffer on
every chip (ttnn.experimental / bringup.high_bw_all_gather): N segments = N histories into every chip and N buffers.

This program (one prepared program a geometry, every layout a set of runtime values):

    home core   (the first core): waits until every OTHER destination chip of its group said "started" (a global
                semaphore; a chip writes into another chip's buffer only once that chip runs this program, so no
                reader of the buffer is still running there), releases its port cores (program semaphore 0), copies
                its own segment's own rows (a destination: local DRAM runs), then waits for the arrivals of every
                other chip of its group (one increment a source and link) and takes them back.
    port cores  (one per fabric direction and link that has a neighbour; its own fabric connection): the root
                direction of a unicast route to its neighbour = its own direction bits (decoded from the packet
                header the fabric's own route tables filled, tt_fabric_api.h: the Y map's byte at this chip's row,
                else the X map's at its column). Link 0 ports of a destination chip send "started" to every chip of
                its group whose route starts in their direction. After the release, for every destination chip d of
                the group (d != this chip) whose route starts in its direction: segment k = map(d); this chip's
                records rows [lo_k, hi_k) of record buffer k -> rows owner(me) R + [lo_k, hi_k) of d's buffer, in
                runs of up to K rows of one DRAM bank (a fabric packet each; the runs of link l: chunk c with
                c % L == l), the last packet of each link with an arrival increment on d (fused), a bare increment
                when the link has no run.

Runtime values (prepared common value slots, the same program for every layout, rule 20): the record buffers'
addresses, the row ranges and the destination map (4 bits a chip: the segment, 0xF = not a destination). Group: the
whole mesh (a TP-dedup cache: owner q = chip-linear r TP + c) or a column (TP-replicated: owner q = r, SP).
Every wait is bounded (SPIN polls, then the core goes on: a hang guard, not a result; the bit-identity tests see it).

Data movement only: every destination chip's rows are the bytes the all-gather put at the same rows of the segment's
lane buffer (records are a function of each served row alone, tt/history_gather._pack)."""

from __future__ import annotations

import ttnn

from .settings import settings

NS = 6  # segment slots a call (tt/packed.GLUED_MAX)
NONE = 0xF  # map value: not a destination
BATCH = 4  # chunks a port reads before sending them (one header each)
SPIN = 1 << 28  # polls of a bounded wait (seconds)

_COMMON = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
using namespace tt::tt_fabric;

constexpr uint32_t PAGE = get_compile_time_arg_val(0);  // bytes a record row (a page of both buffers)
constexpr uint32_t K = get_compile_time_arg_val(1);     // rows a packet at most
constexpr uint32_t N = get_compile_time_arg_val(2);     // chips
constexpr uint32_t C = get_compile_time_arg_val(3);     // mesh columns
constexpr uint32_t AXIS = get_compile_time_arg_val(4);  // 0: the whole mesh is one group, 1: a column
constexpr uint32_t L = get_compile_time_arg_val(5);     // links (port cores) a direction
constexpr uint32_t NSEG = get_compile_time_arg_val(6);  // segment slots
constexpr uint32_t NW = get_compile_time_arg_val(7);    // destination-map words (8 chips a word)
constexpr uint32_t R = get_compile_time_arg_val(8);     // rows an owner slot of the history buffer
constexpr uint32_t B = get_compile_time_arg_val(9);     // chunks a batch
constexpr uint32_t SPIN = get_compile_time_arg_val(10);
constexpr auto o_args = TensorAccessorArgs<11>();
constexpr auto l_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();

// common args (per-run values): [0, NSEG) record buffers, [NSEG, 2 NSEG) lo, [2 NSEG, 3 NSEG) hi, then NW map words
FORCE_INLINE uint32_t seg_of(uint32_t d) {
    return (get_common_arg_val<uint32_t>(3 * NSEG + d / 8) >> (4 * (d % 8))) & 0xFu;
}
FORCE_INLINE bool in_group(uint32_t d, uint32_t me) { return AXIS == 0 || (d % C) == (me % C); }
FORCE_INLINE uint32_t owner(uint32_t d) { return AXIS == 0 ? d : d / C; }
constexpr uint32_t GROUP = AXIS == 0 ? N : N / C;

FORCE_INLINE void wait_min(uint32_t addr, uint32_t v) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t i = 0; i < SPIN && *p < v; ++i) {
        invalidate_l1_cache();
    }
}

FORCE_INLINE void take_back(uint32_t addr, uint32_t v) {
    noc_semaphore_inc(get_noc_addr(addr), 0u - v);
    noc_async_atomic_barrier();
}

// DRAM banks of an interleaved buffer: page j is in the bank of page 0 again at j = banks (probe, <= 64)
template <typename A>
FORCE_INLINE uint32_t banks_of(const A& acc) {
    const uint64_t a0 = acc.get_noc_addr(0);
    for (uint32_t j = 1; j <= 64; ++j) {
        if (acc.get_noc_addr(j) == a0 + PAGE) {
            return j;
        }
    }
    return 1;
}

// runs of rows [lo, hi): blocks of nb K rows from lo; in a block, bank class b's rows first, first + nb, ... (one
// DRAM bank, contiguous in both buffers). Chunk c = block c / nb, class c % nb.
FORCE_INLINE uint32_t n_chunks(uint32_t lo, uint32_t hi, uint32_t nb) {
    if (hi <= lo) {
        return 0;
    }
    const uint32_t full = (hi - lo) / (nb * K), rem = (hi - lo) - full * nb * K;
    return full * nb + (rem < nb ? rem : nb);
}
FORCE_INLINE void chunk_at(uint32_t c, uint32_t lo, uint32_t hi, uint32_t nb, uint32_t& first, uint32_t& cnt) {
    const uint32_t i0 = lo + (c / nb) * nb * K;
    const uint32_t end = i0 + nb * K < hi ? i0 + nb * K : hi;
    first = i0 + c % nb;
    cnt = (end - first + nb - 1) / nb;
}
"""

_HOME = (
    _COMMON
    + r"""
void kernel_main() {
    // [out, me, started, arrived, nports, (px, py) x nports]
    const uint32_t out = get_arg_val<uint32_t>(0);
    const uint32_t me = get_arg_val<uint32_t>(1);
    const uint32_t started = get_arg_val<uint32_t>(2);
    const uint32_t arrived = get_arg_val<uint32_t>(3);
    const uint32_t np = get_arg_val<uint32_t>(4);
    uint32_t ndest = 0;
    for (uint32_t d = 0; d < N; ++d) {
        if (d != me && in_group(d, me) && seg_of(d) != 0xFu) {
            ++ndest;
        }
    }
    if (ndest) {  // every other destination of the group runs this program: its buffer is free to write
        wait_min(started, ndest);
        take_back(started, ndest);
    }
    for (uint32_t p = 0; p < np; ++p) {
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(5 + 2 * p), get_arg_val<uint32_t>(6 + 2 * p),
                                       get_semaphore(0)), 1);
    }
    noc_async_atomic_barrier();
    const uint32_t k = seg_of(me);
    if (k == 0xFu) {
        return;
    }
    const uint32_t lo = get_common_arg_val<uint32_t>(NSEG + k);
    const uint32_t hi = get_common_arg_val<uint32_t>(2 * NSEG + k);
    const auto O = TensorAccessor(o_args, out, PAGE);
    const auto S = TensorAccessor(l_args, get_common_arg_val<uint32_t>(k), PAGE);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    if (hi > lo) {  // the own rows: local runs (read, then write; one run at a time per slot)
        const uint32_t nb = banks_of(O);
        const uint32_t base = owner(me) * R;
        const uint32_t total = n_chunks(lo, hi, nb);
        for (uint32_t c0 = 0; c0 < total; c0 += B) {
            const uint32_t m = total - c0 < B ? total - c0 : B;
            uint32_t first, cnt;
            for (uint32_t i = 0; i < m; ++i) {
                chunk_at(c0 + i, lo, hi, nb, first, cnt);
                noc_async_read(S.get_noc_addr(first), buf + i * K * PAGE, cnt * PAGE);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < m; ++i) {
                chunk_at(c0 + i, lo, hi, nb, first, cnt);
                noc_async_write(buf + i * K * PAGE, O.get_noc_addr(base + first), cnt * PAGE);
            }
            noc_async_write_barrier();
        }
    }
    if (GROUP > 1) {  // every other chip of the group: one increment a link
        wait_min(arrived, (GROUP - 1) * L);
        take_back(arrived, (GROUP - 1) * L);
    }
}
"""
)

_PORT = (
    _COMMON
    + r"""
FORCE_INLINE uint8_t root_bits(volatile tt_l1_ptr PACKET_HEADER_TYPE* h) {
    tt_l1_ptr routing_l1_info_t* rt = reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const uint8_t ay = h->route_buffer[rt->my_mesh_coord_y];
    const uint8_t a = ay != 0 ? ay : h->route_buffer[rt->mesh_y_size + rt->my_mesh_coord_x];
    return a & Routing2DCodec::ACTION_ETH_MASK;
}

void kernel_main() {
    // [out, me, peer chip, mesh id, link, hx, hy, started, arrived, chip id x N, conn..]
    size_t a = 0;
    const uint32_t out = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);
    const uint16_t peer = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t link = get_arg_val<uint32_t>(a++);
    const uint32_t hx = get_arg_val<uint32_t>(a++);
    const uint32_t hy = get_arg_val<uint32_t>(a++);
    const uint32_t started = get_arg_val<uint32_t>(a++);
    const uint32_t arrived = get_arg_val<uint32_t>(a++);
    const uint32_t chips = a;
    a += N;
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
#if !(defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0))
#error "the owned history send needs a 2D fabric (agent rule 4)"
#endif
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr[2 * B];
    for (uint32_t i = 0; i < 2 * B; ++i) {
        hdr[i] = PacketHeaderPool::allocate_header();
    }
    (void)fabric_set_unicast_route(hdr[0], peer, mesh_id);
    const uint8_t mine = root_bits(hdr[0]);
    auto route = [&](uint32_t d) -> bool {  // every header to chip d; true when its route starts here
        for (uint32_t i = 0; i < 2 * B; ++i) {
            (void)fabric_set_unicast_route(hdr[i], static_cast<uint16_t>(get_arg_val<uint32_t>(chips + d)), mesh_id);
        }
        return root_bits(hdr[0]) == mine;
    };
    auto bare_inc = [&](uint64_t sem) {
        hdr[0]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{sem, 1, true});
        conn.wait_for_empty_write_slot();
        conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr[0]), sizeof(PACKET_HEADER_TYPE));
        noc_async_writes_flushed();
    };
    conn.open();
    // 1. "started" to every chip of the group whose route starts here (link 0, when this chip is a destination)
    if (link == 0 && seg_of(me) != 0xFu) {
        for (uint32_t s = 0; s < N; ++s) {
            if (s == me || !in_group(s, me) || !route(s)) {
                continue;
            }
            bare_inc(get_noc_addr(hx, hy, started));
        }
    }
    // 2. released by the home core once every destination started
    wait_min(get_semaphore(0), 1);
    // 3. the runs of this link to every destination whose route starts here
    const auto O = TensorAccessor(o_args, out, PAGE);
    const uint32_t nb = banks_of(O);
    const uint32_t base = owner(me) * R;
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    for (uint32_t d = 0; d < N; ++d) {
        const uint32_t k = seg_of(d);
        if (d == me || !in_group(d, me) || k == 0xFu || !route(d)) {
            continue;
        }
        const uint32_t lo = get_common_arg_val<uint32_t>(NSEG + k);
        const uint32_t hi = get_common_arg_val<uint32_t>(2 * NSEG + k);
        const auto S = TensorAccessor(l_args, get_common_arg_val<uint32_t>(k), PAGE);
        const uint64_t arrival = get_noc_addr(hx, hy, arrived);
        const uint32_t total = n_chunks(lo, hi, nb);
        if (total <= link) {  // no run of this link: still one increment
            bare_inc(arrival);
            continue;
        }
        const uint32_t mine_n = (total - 1 - link) / L + 1;  // this link's chunks: link, link + L, ...
        // two halves of B slots: the next half's DRAM reads run while this half's packets go out
        auto fill = [&](uint32_t half, uint32_t j0) {
            const uint32_t m = mine_n - j0 < B ? mine_n - j0 : B;
            uint32_t first, cnt;
            for (uint32_t i = 0; i < m; ++i) {
                chunk_at(link + (j0 + i) * L, lo, hi, nb, first, cnt);
                noc_async_read(S.get_noc_addr(first), buf + (half * B + i) * K * PAGE, cnt * PAGE);
            }
        };
        uint32_t half = 0;
        fill(0, 0);
        for (uint32_t j0 = 0; j0 < mine_n; j0 += B) {
            const uint32_t m = mine_n - j0 < B ? mine_n - j0 : B;
            noc_async_read_barrier();
            noc_async_writes_flushed();  // (the other half's packets have left L1: its slots and headers are free)
            for (uint32_t i = 0; i < m; ++i) {
                uint32_t first, cnt;
                chunk_at(link + (j0 + i) * L, lo, hi, nb, first, cnt);
                const uint64_t dst = tt::tt_fabric::linear::addrgen_detail::get_noc_address(O, base + first);
                const uint32_t bytes = cnt * PAGE;
                volatile tt_l1_ptr PACKET_HEADER_TYPE* h = hdr[half * B + i];
                if (j0 + i + 1 == mine_n) {
                    h->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{dst, arrival, 1, true}, bytes);
                } else {
                    h->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(
                    buf + (half * B + i) * K * PAGE, bytes, reinterpret_cast<uint32_t>(h));
            }
            if (j0 + B < mine_n) {
                fill(half ^ 1, j0 + B);
            }
            half ^= 1;
        }
        noc_async_writes_flushed();
    }
    conn.close();
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), 0);
}
"""
)


def _geom(t: ttnn.Tensor) -> tuple:
    return (tuple(t.shape), str(t.dtype), str(t.layout), str(t.memory_config()))


def links() -> int:
    return int(settings.get("HIST_OWNED_LINKS"))


class OwnedSend:
    """The owned history send of one mesh (module docstring). ``__call__(out, R, axis, entries, dest)``."""

    def __init__(self, mesh, nlinks: int):
        from ttnn.bringup.combine_rf.sparse_all_gather_2d import mesh_plan

        from .window_send import _torus

        self.mesh = mesh
        self.R_mesh, self.C = tuple(mesh.shape)
        self.n = self.R_mesh * self.C
        assert 1 < self.n <= 8 * 8, self.n
        self.L = int(nlinks)
        self.chips, _, _ = mesh_plan(mesh, self.L, _torus(mesh))
        payload = min(int(ttnn.get_tt_fabric_max_payload_size_bytes()), 6208)  # (owner: fabric payloads <= 6208 B)
        self.payload = payload
        g = mesh.compute_with_storage_grid_size()
        nports = max(len(p) for p in self.chips.values()) * self.L
        need = 1 + nports
        x = int(g.x) - 1  # (the gather column: tt/gather_hide.Split's sub-device 1 on the Galaxy)
        assert need <= int(g.y), f"owned send: {need} cores, the column has {g.y}"
        self.cores = [ttnn.CoreCoord(x, y) for y in range(need)]
        hc = self.cores[0]
        hcrs = ttnn.CoreRangeSet([ttnn.CoreRange(hc, hc)])
        # (load time: the started / arrived counters on the home core, left at 0 after every call)
        self.sems = [ttnn.create_global_semaphore(mesh, hcrs, 0, ttnn.BufferType.L1_SMALL) for _ in range(2)]
        self.sem_addr = [int(ttnn.get_global_semaphore_address(s)) for s in self.sems]
        v = mesh.worker_core_from_logical_core(hc)
        self.home_xy = (int(v.x), int(v.y))
        self.port_xy = []
        for c in self.cores[1:]:
            w = mesh.worker_core_from_logical_core(c)
            self.port_xy.append((int(w.x), int(w.y)))
        self.nw = -(-self.n // 8)
        self._pp = {}

    def _node(self, ch):
        return self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(*ch))

    def _build(self, out: ttnn.Tensor, loc: ttnn.Tensor, R: int, axis: int, page: int, k: int):
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        ct = [page, k, self.n, self.C, axis, self.L, NS, self.nw, R, BATCH, SPIN] + acc(out) + acc(loc)
        common = [0] * (3 * NS + self.nw)
        oa = int(out.buffer_address())
        dm = ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
        )
        cb_bytes = 2 * BATCH * k * page + 64
        ids = [int(self._node(divmod(q, self.C)).chip_id) for q in range(self.n)]
        md = ttnn.MeshProgramDescriptor()
        hc = self.cores[0]
        n_ports = 0
        for r in range(self.R_mesh):
            for c in range(self.C):
                me = r * self.C + c
                ports = []  # (core, peer, link)
                for d in sorted(self.chips[(r, c)]):
                    p = self.chips[(r, c)][d]
                    for li in range(self.L):
                        ports.append((self.cores[1 + len(ports)], tuple(p["peer"]), p["links"][li], li))
                n_ports += len(ports)
                pcs = [pc for pc, _, _, _ in ports]
                all_crs = ttnn.CoreRangeSet([ttnn.CoreRange(x, x) for x in [hc] + pcs])
                hcrs = ttnn.CoreRangeSet([ttnn.CoreRange(hc, hc)])
                pcrs = ttnn.CoreRangeSet([ttnn.CoreRange(x, x) for x in pcs])
                prog = ttnn.ProgramDescriptor()
                prog.cbs = [
                    ttnn.CBDescriptor(
                        total_size=cb_bytes,
                        core_ranges=all_crs,
                        format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, cb_bytes)],
                    )
                ]
                prog.semaphores = [ttnn.SemaphoreDescriptor(id=0, core_ranges=all_crs, initial_value=0)]
                hrt = ttnn.RuntimeArgs()
                hargs = [oa, me, self.sem_addr[0], self.sem_addr[1], len(ports)]
                for i in range(len(ports)):
                    hargs += list(self.port_xy[i])
                hrt[hc.x][hc.y] = hargs
                prt = ttnn.RuntimeArgs()
                for pc, peer, link_id, li in ports:
                    pn = self._node(peer)
                    args = [oa, me, int(pn.chip_id), int(pn.mesh_id), li, self.home_xy[0], self.home_xy[1]]
                    args += list(self.sem_addr) + ids
                    args += list(ttnn.setup_fabric_connection(self.mesh, self._node((r, c)), pn, link_id, prog, pc))
                    prt[pc.x][pc.y] = args
                prog.kernels = [
                    ttnn.KernelDescriptor(
                        kernel_source=_HOME,
                        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                        core_ranges=hcrs,
                        compile_time_args=ct,
                        runtime_args=hrt,
                        common_runtime_args=list(common),
                        config=dm,
                    ),
                    ttnn.KernelDescriptor(
                        kernel_source=_PORT,
                        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                        core_ranges=pcrs,
                        compile_time_args=ct,
                        runtime_args=prt,
                        common_runtime_args=list(common),
                        config=dm,
                    ),
                ]
                md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = prog
        return md, n_ports

    def __call__(self, out: ttnn.Tensor, R: int, axis: int, entries: list, dest: list) -> None:
        """``out`` [1, 1, owners R, page] uint8 ROW_MAJOR DRAM interleaved (the source's history buffer);
        ``entries`` [(record buffer [1, 1, R, page], lo, hi)] (<= NS; rows [lo, hi) of every owner); ``dest`` per chip
        (chip-linear): the entry index it receives, or None. ``axis`` 0: the group is the whole mesh, 1: a column."""
        assert 0 < len(entries) <= NS and len(dest) == self.n, (len(entries), len(dest))
        page = int(out.shape[-1])
        loc0 = next(t for t, _, _ in entries if t is not None)
        assert page % 64 == 0 and int(loc0.shape[-1]) == page and int(loc0.shape[-2]) == R, (out.shape, loc0.shape)
        k = max(1, self.payload // page)
        key = (_geom(out), _geom(loc0), int(R), int(axis))
        hit = self._pp.get(key)
        if hit is None:
            md, _ = self._build(out, loc0, int(R), int(axis), page, k)
            nv = 3 * NS + self.nw
            slots = [(kern, (1 << 31) | i) for kern in (0, 1) for i in range(nv)]
            hit = self._pp[key] = ttnn.bringup.prepare_program(md, [out], slots)
        pp = hit
        locs, los, his = [0] * NS, [0] * NS, [0] * NS
        for j, (t, lo, hi) in enumerate(entries):
            if t is None:
                continue
            assert _geom(t) == key[1], ("owned send: one record-buffer geometry a call", _geom(t), key[1])
            assert 0 <= int(lo) <= int(hi) <= R, (lo, hi, R)
            locs[j], los[j], his[j] = int(t.buffer_address()), int(lo), int(hi)
        words = [0] * self.nw
        for q, e in enumerate(dest):
            v = NONE if e is None else int(e)
            assert v == NONE or 0 <= v < len(entries), (q, e)
            words[q // 8] |= v << (4 * (q % 8))
        vals = locs + los + his + words
        ttnn.bringup.run_prepared(pp, [out], vals * 2)


_SEND: dict = {}


def sender(mesh) -> OwnedSend:
    nl = links()
    hit = _SEND.get(id(mesh))
    if hit is None or hit[0] is not mesh or hit[1].L != nl:
        hit = _SEND[id(mesh)] = (mesh, OwnedSend(mesh, nl))
    return hit[1]
