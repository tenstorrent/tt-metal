# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""tail_a2a - the fused tail of a model without a latent (GLM, K2.7, M3, ...) as one program a chip: out [S/SP, H/TP]
(the width split, every token of the mesh row) = the TP all-to-all of z = tilize(y) + se, with y the combine's
row-major owner rows [S/N, H] and se the shared expert's owner rows [S/N, H] (TILE) - the ops tail's tilize, add and
all_to_all_async_generic in one program, the add and tilize under the sends.
  workers  every core but the ports, NW = H / 32 / w of them: worker i owns tile columns [i w, i w + w) (inside one
           destination block) of every own row tile. Per row tile: the 32 rows' slice (row-major, NoC0 reads), tilized,
           + se's tiles, written - its own block's straight into the output, another chip's into the staging buffer
           z [S/N, H] (DRAM, TILE) - then +1 on every carrying port's z-ready word of that row tile (NoC1; a word a
           row tile: the workers run at their own pace - one count over the call let fast workers' later row tiles
           stand in for a slow one's, a port read a row tile before it was written)
  ports    a core per (direction, link) (fabric_all_gather's plan: below the link's Ethernet core). Reader: per row
           tile, once every worker wrote it (its z-ready word at NW), the pages of each
           destination block it carries - its link's banks, runs of pages one bank apart (consecutive in the bank, in
           z and in the destination's output). Sender: each run into the destination's output pages (a 2D unicast,
           multi-hop), every inc_every-th run of a destination (and its last) a fused write + increment of the
           destination's arrival word (on its port facing me)
  fence    every port tells each chip that sends into my output through it "ready" (a fabric increment of that
           sender's ready word, at program start: whatever read my previous output has run), and sends nothing before
           every destination's ready; at its end it waits for every arrival it expects. Words are re-armed
           (decremented by what was consumed): trace-safe, nothing to reset between calls
  links    line: a destination's block travels toward it (the middle links of a line of 4 carry 4 blocks: the floor);
           ring: the shorter way, at G / 2 (G even) the way the fabric's own routing sends it (XGather3.ring_dir: a
           2D unicast injected toward the other neighbour is not delivered - the first ring run hung on it). A ready
           goes the fabric's way back to its sender: on a tie that need not be the side the data arrives on."""

import os

import torch
from ttnn.bringup.combine_rf import fabric_all_gather_py as fag
from ttnn.bringup.prepared_program import generic_op as prepared_generic_op

import ttnn
from ttnn.bringup.combine_rf import persist_l1
from models.demos.moe_block.tt.se_engine3 import _ring_pages

CB_RM, CB_T, CB_SE, CB_OUT, CB_CH = 0, 1, 2, 3, 4
INC_EVERY = 8

_ROUTE = r"""
// 2D fabrics route by destination node (multi-hop); 1D by hop count
inline void route_to(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id, uint32_t hops) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)) || defined(EMULE_FABRIC_2D)
    (void)hops;
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
    (void)chip_id;
    (void)mesh_id;
    (void)fabric_set_unicast_route<false>(
        reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(hdr), static_cast<uint16_t>(hops));
#endif
}
"""

# a port's runs of a destination block's row tile: link l of L takes the bank offsets t0 = l, l + L, .. (< min(B, NTB));
# part 1 / 2 the first / second half of those; a run = up to RP pages t0 + (m + i) B
_WALK = r"""
template <typename F>
FORCE_INLINE void for_each_run(uint32_t NTB, uint32_t l, uint32_t L, uint32_t part, F&& f) {
    const uint32_t nt = NTB < NB ? NTB : NB;
    const uint32_t nl = nt > l ? (nt - l + L - 1) / L : 0;  // (my bank offsets)
    const uint32_t lo = part == 2 ? nl / 2 : 0, hi = part == 1 ? nl / 2 : nl;
    for (uint32_t i = lo; i < hi; ++i) {
        const uint32_t t0 = l + i * L;
        for (uint32_t m = t0; m < NTB; m += RP * NB) {
            const uint32_t left = (NTB - m + NB - 1) / NB;
            f(m, left < RP ? left : RP);
        }
    }
}
"""

_PORT_HDR = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t CB = get_compile_time_arg_val(0), PB = get_compile_time_arg_val(1), RP = get_compile_time_arg_val(2);
constexpr uint32_t NB = get_compile_time_arg_val(3), GROUP = get_compile_time_arg_val(4), INC = get_compile_time_arg_val(5);
"""

_PORT_READER = (
    _PORT_HDR
    + _WALK
    + r"""
// my runs of every destination, row tile by row tile, from z once its workers wrote the row tile
void kernel_main() {
    constexpr auto z_args = TensorAccessorArgs<6>();
    size_t a = 0;
    const uint32_t z_addr = get_arg_val<uint32_t>(a++), NW = get_arg_val<uint32_t>(a++);
    const uint32_t zr_addr = get_arg_val<uint32_t>(a++), RT = get_arg_val<uint32_t>(a++);
    const uint32_t NTH = get_arg_val<uint32_t>(a++), NTB = get_arg_val<uint32_t>(a++);
    const uint32_t l = get_arg_val<uint32_t>(a++), L = get_arg_val<uint32_t>(a++), nd = get_arg_val<uint32_t>(a++);
    const uint32_t d0 = a;
    const auto z = TensorAccessor(z_args, z_addr, PB);
    volatile tt_l1_ptr uint32_t* zr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zr_addr);
    if (nd == 0) return;
    uint32_t batch = 0, wptr = 0;
    for (uint32_t r = 0; r < RT; ++r) {
        noc_semaphore_wait_min(zr + 4 * r, NW);  // (every worker wrote row tile r: its word, 16 B apart)
        for (uint32_t k = 0; k < nd; ++k) {
            const uint32_t e = get_arg_val<uint32_t>(d0 + k), q = e & 0xFF, part = (e >> 8) & 3;
            const uint32_t base = r * NTH + q * NTB;
            for_each_run(NTB, l, L, part, [&](uint32_t m, uint32_t n) {
                if (batch == 0) {
                    cb_reserve_back(CB, RP * GROUP);
                    wptr = get_write_ptr(CB);
                }
                for (uint32_t i = 0; i < n; ++i) {  // (z's pages m, m + NB, ..: one bank, consecutive)
                    noc_async_read(z.get_noc_addr(base + m + i * NB), wptr + (batch * RP + i) * PB, PB);
                }
                if (++batch == GROUP) {
                    noc_async_read_barrier();
                    cb_push_back(CB, RP * GROUP);
                    batch = 0;
                }
            });
        }
    }
    if (batch) {
        noc_async_read_barrier();
        cb_push_back(CB, RP * batch);
    }
    for (uint32_t r = 0; r < RT; ++r) {  // (re-arm: every increment of the call landed)
        noc_semaphore_inc(get_noc_addr(zr_addr + 16 * r), 0u - NW);
    }
    noc_async_atomic_barrier();
}
"""
)

_PORT_SENDER = (
    _PORT_HDR
    + r"""
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
"""
    + _ROUTE
    + _WALK
    + r"""
constexpr uint32_t MAXD = 8;
// readies to the chips that send into my output through me; every destination's ready awaited; my runs into the
// destinations' output pages; at the end every arrival I expect
void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<6>();
    size_t a = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(a++), me = get_arg_val<uint32_t>(a++), RT = get_arg_val<uint32_t>(a++);
    const uint32_t NTB = get_arg_val<uint32_t>(a++), l = get_arg_val<uint32_t>(a++), L = get_arg_val<uint32_t>(a++);
    const uint32_t arr_addr = get_arg_val<uint32_t>(a++), expect_in = get_arg_val<uint32_t>(a++);
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++), nready = get_arg_val<uint32_t>(a++);
    const uint32_t nrs = get_arg_val<uint32_t>(a++);  // (readies I send: x | y << 16, chip, mesh, hops)
    const uint32_t rs0 = a;
    a += 4 * nrs;
    const uint32_t nd = get_arg_val<uint32_t>(a++);  // (destinations: q | part << 8, chip, mesh, hops, runs, x | y << 16)
    const uint32_t d0 = a;
    a += 6 * nd;
    const auto out = TensorAccessor(out_args, out_addr, PB);
    if (nd > 0 || nrs > 0) {
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);  // (appended last)
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[GROUP];
        for (uint32_t h = 0; h < GROUP; ++h) hdrs[h] = PacketHeaderPool::allocate_header();
        conn.open();
        for (uint32_t i = 0; i < nrs; ++i) {  // (my output may be written: say so to each of its senders here)
            const uint32_t xy = get_arg_val<uint32_t>(rs0 + 4 * i);
            route_to(hdrs[0], get_arg_val<uint32_t>(rs0 + 4 * i + 1), get_arg_val<uint32_t>(rs0 + 4 * i + 2),
                     get_arg_val<uint32_t>(rs0 + 4 * i + 3));
            hdrs[0]->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{get_noc_addr(xy & 0xFFFF, xy >> 16, ready_addr), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdrs[0]), sizeof(PACKET_HEADER_TYPE));
        }
        if (nd > 0) {
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            noc_semaphore_wait_min(ready, nready);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - nready);
            noc_async_atomic_barrier();
        }
        uint32_t sent[MAXD] = {};
        uint32_t h = 0, unflushed = 0;
        for (uint32_t r = 0; r < RT; ++r) {
            for (uint32_t k = 0; k < nd; ++k) {
                const uint32_t e = get_arg_val<uint32_t>(d0 + 6 * k), part = (e >> 8) & 3;
                const uint16_t chip = get_arg_val<uint32_t>(d0 + 6 * k + 1), mesh = get_arg_val<uint32_t>(d0 + 6 * k + 2);
                const uint32_t hops = get_arg_val<uint32_t>(d0 + 6 * k + 3), runs = get_arg_val<uint32_t>(d0 + 6 * k + 4);
                const uint32_t pxy = get_arg_val<uint32_t>(d0 + 6 * k + 5);
                const uint64_t arr = get_noc_addr(pxy & 0xFFFF, pxy >> 16, arr_addr);
                const uint32_t base = (me * RT + r) * NTB;  // (my rows' row tile r in the destination's output)
                for_each_run(NTB, l, L, part, [&](uint32_t m, uint32_t n) {
                    cb_wait_front(CB, RP * (unflushed + 1));
                    const uint32_t src = get_read_ptr(CB) + unflushed * RP * PB;
                    const uint64_t dst = out.get_noc_addr(base + m, 0, 0);  // (pages m, m + NB, ..: consecutive)
                    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                    h = h + 1 == GROUP ? 0 : h + 1;
                    route_to(hdr, chip, mesh, hops);
                    if (++sent[k] % INC == 0 || sent[k] == runs) {
                        hdr->to_noc_fused_unicast_write_atomic_inc(NocUnicastAtomicIncFusedCommandHeader{dst, arr, 1, true}, n * PB);
                    } else {
                        hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, n * PB);
                    }
                    conn.wait_for_empty_write_slot();
                    conn.send_current_slot_non_blocking(src, n * PB, reinterpret_cast<uint32_t>(hdr));
                    if (++unflushed == GROUP) {
                        noc_async_writes_flushed();
                        cb_pop_front(CB, RP * GROUP);
                        unflushed = 0;
                    }
                });
            }
        }
        if (unflushed) {
            noc_async_writes_flushed();
            cb_pop_front(CB, RP * unflushed);
        }
        conn.close();
    }
    if (expect_in > 0) {  // (everything sent into my output through my side landed; then re-armed)
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arr_addr);
        noc_semaphore_wait_min(arrived, expect_in);
        noc_semaphore_inc(get_noc_addr(arr_addr), 0u - expect_in);
        noc_async_atomic_barrier();
    }
}
"""
)

_WORKER_READER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
// my column slice of the owners' rows (row-major, 32 rows a row tile) and se's tiles, row tile by row tile
void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0), SE = get_compile_time_arg_val(1);
    constexpr auto y_args = TensorAccessorArgs<2>();
    constexpr auto s_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    const uint32_t y_addr = get_arg_val<uint32_t>(0), s_addr = get_arg_val<uint32_t>(1), c0 = get_arg_val<uint32_t>(2);
    const uint32_t RT = get_arg_val<uint32_t>(3), NTH = get_arg_val<uint32_t>(4);
    const auto y = TensorAccessor(y_args, y_addr, NTH * 64);
    const auto s = TensorAccessor(s_args, s_addr, 2048);
    for (uint32_t r = 0; r < RT; ++r) {
        cb_reserve_back(0, W);
        const uint32_t l1 = get_write_ptr(0);
        for (uint32_t i = 0; i < 32; ++i) {
            noc_async_read(y.get_noc_addr(r * 32 + i, c0 * 64), l1 + i * W * 64, W * 64);
        }
        if constexpr (SE) {
            cb_reserve_back(2, W);
            const uint32_t s1 = get_write_ptr(2);
            for (uint32_t t = 0; t < W; ++t) noc_async_read(s.get_noc_addr(r * NTH + c0 + t), s1 + t * 2048, 2048);
        }
        noc_async_read_barrier();
        cb_push_back(0, W);
        if constexpr (SE) cb_push_back(2, W);
    }
}
"""

_WORKER_COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/tilize.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
// tilize my slice; + se's tiles
void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0), SE = get_compile_time_arg_val(1);
    const uint32_t RT = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0, 2, 3);
    for (uint32_t r = 0; r < RT; ++r) {
        const uint32_t tcb = SE ? 1 : 3;
        cb_wait_front(0, W);
        cb_reserve_back(tcb, W);
        reconfig_data_format_srca(0);
        pack_reconfig_data_format(tcb);
        tilize_init(0, W, tcb);
        tilize_block(0, W, tcb);
        tilize_uninit(0, tcb);
        cb_push_back(tcb, W);
        cb_pop_front(0, W);
        if constexpr (SE) {
            cb_wait_front(1, W);
            cb_wait_front(2, W);
            cb_reserve_back(3, W);
            reconfig_data_format(1, 2);
            pack_reconfig_data_format(3);
            add_init(1, 2);
            tile_regs_acquire();
            for (uint32_t t = 0; t < W; ++t) add_tiles(1, 2, t, t, t);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < W; ++t) pack_tile(t, 3);
            tile_regs_release();
            cb_push_back(3, W);
            cb_pop_front(1, W);
            cb_pop_front(2, W);
        }
    }
}
"""

_WORKER_WRITER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
// my tiles: my own block's into the output, another chip's into z; then +1 on every port's z-ready word
void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);
    constexpr auto o_args = TensorAccessorArgs<1>();
    constexpr auto z_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    size_t a = 0;
    const uint32_t o_addr = get_arg_val<uint32_t>(a++), z_addr = get_arg_val<uint32_t>(a++), c0 = get_arg_val<uint32_t>(a++);
    const uint32_t RT = get_arg_val<uint32_t>(a++), NTH = get_arg_val<uint32_t>(a++), NTB = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++), zr_addr = get_arg_val<uint32_t>(a++), np = get_arg_val<uint32_t>(a++);
    const uint32_t p0 = a;
    const auto o = TensorAccessor(o_args, o_addr, 2048);
    const auto z = TensorAccessor(z_args, z_addr, 2048);
    const uint32_t q = c0 / NTB;
    for (uint32_t r = 0; r < RT; ++r) {
        cb_wait_front(3, W);
        const uint32_t l1 = get_read_ptr(3);
        for (uint32_t t = 0; t < W; ++t) {
            const uint64_t dst = q == me ? o.get_noc_addr((me * RT + r) * NTB + c0 - q * NTB + t)
                                         : z.get_noc_addr(r * NTH + c0 + t);
            noc_async_write(l1 + t * 2048, dst, 2048);
        }
        noc_async_write_barrier();
        cb_pop_front(3, W);
        for (uint32_t i = 0; i < np; ++i) {
            const uint32_t xy = get_arg_val<uint32_t>(p0 + i);
            noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, zr_addr + 16 * r), 1);
        }
    }
    noc_async_atomic_barrier();
}
"""


class TailSchedule:
    """Who sends what to whom (host; the device program and the protocol model share it): G chips of a TP group at
    positions 0 .. G - 1 (a line, or a ring), L links, a destination block NTB tiles wide, RT own row tiles a chip,
    NBK DRAM banks, RP pages a run. Ports are (position, direction, link)."""

    def __init__(self, *, G, ring, L, NTB, RT, NBK, RP, ties=None):
        self.G, self.ring, self.L, self.NTB, self.RT, self.NBK, self.RP = G, ring, L, NTB, RT, NBK, RP
        self.ties = ties or {}  # ((p, q) at distance G / 2 on a ring -> the fabric's direction; default "fwd")
        inc = lambda n: -(-n // INC_EVERY)  # noqa: E731  (an increment every INC_EVERY runs and the last)
        self.expect, self.readies = {}, {}
        for p in range(G):
            for d in ("fwd", "bwd"):
                for l in range(L):
                    for q, part in self.dests(p, d):
                        # (my arrivals: the destination's port facing the side my packets come in by; my ready: its
                        # port facing the fabric's way back to me)
                        akey = (q, fag._OPP[d], l)
                        self.expect[akey] = self.expect.get(akey, 0) + inc(RT * self.runs(part, l))
                        self.readies.setdefault((q, self.dir(q, p), l), set()).add((p, d, l))

    def dir(self, p, q):
        """The way a packet from position p to q leaves: toward it on a line; the shorter way on a ring, a tie the
        fabric's."""
        if not self.ring:
            return "fwd" if q > p else "bwd"
        df = (q - p) % self.G
        if 2 * df == self.G:
            return self.ties.get((p, q), "fwd")
        return "fwd" if 2 * df < self.G else "bwd"

    def dests(self, p, d):
        """[(q, part)] position p sends toward d, nearest first (part 0: all its runs; parts 1 / 2 - halves - unused:
        a tie goes one way)."""
        G = self.G
        if not self.ring:
            return [(q, 0) for q in (range(p + 1, G) if d == "fwd" else range(p - 1, -1, -1))]
        qs = [((p + k) % G if d == "fwd" else (p - k) % G) for k in range(1, G)]
        return [(q, 0) for q in qs if self.dir(p, q) == d]

    def run_list(self, part, l):
        """A port of link l's runs of a destination's row tile: [(first tile m, pages n)] (for_each_run)."""
        nt = min(self.NBK, self.NTB)
        nl = len(range(l, nt, self.L))
        lo, hi = (nl // 2 if part == 2 else 0), (nl // 2 if part == 1 else nl)
        out = []
        for i in range(lo, hi):
            t0 = l + i * self.L
            for m in range(t0, self.NTB, self.RP * self.NBK):
                out.append((m, min(self.RP, -(-(self.NTB - m) // self.NBK))))
        return out

    def runs(self, part, l):
        return len(self.run_list(part, l))

    def hops(self, p, q, d):
        return ((q - p) if d == "fwd" else (p - q)) % self.G


class TailA2A:
    """The non-latent fused tail (module docstring): rows - a mesh row's tokens (S/SP), H, the TP collective's topology
    and links. run(y_rm, se) -> [1, 1, rows, H / TP] TILE, DRAM."""

    def __init__(self, mesh, *, rows, H, topology, num_links, w_max=8):
        R, G = tuple(mesh.shape)
        self.mesh, self.G, self.rows, self.H, self.L = mesh, G, rows, H, num_links
        self.S = rows // G  # (owned rows a chip)
        assert self.S % 32 == 0 and H % (32 * G) == 0, f"tail_a2a: {rows} rows, H {H} over TP {G}: whole tiles"
        self.RT, self.NTH, self.NTB = self.S // 32, H // 32, H // 32 // G
        self.W = max(d for d in range(1, w_max + 1) if self.NTB % d == 0)
        self.NW = self.NTH // self.W
        self.chips, self.ring = fag.plan(mesh, cluster_axis=1, topology=topology, num_links=num_links)
        self.NBK = fag._num_banks(mesh)
        self.page = 2048
        self.RP = max(1, ttnn.get_tt_fabric_max_payload_size_bytes() // self.page)
        self.group = max(1, min(8, (96 * 1024) // (2 * self.RP * self.page)))
        grid = mesh.compute_with_storage_grid_size()
        ports = {(c.x, c.y) for ch in self.chips.values() for c in ch["ports"].values()}
        free = [(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in ports]
        assert len(free) >= self.NW, f"tail_a2a: {self.NW} workers, {len(free)} free cores"
        self.workers = [ttnn.CoreCoord(x, y) for x, y in free[: self.NW]]
        pc = sorted(ports)
        self.port_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in pc])
        self.work_crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.workers])
        self.sems = [persist_l1.semaphore(mesh, self.port_crs, 0) for _ in range(2)]  # (arrival, ready)
        # (the z-ready words: a row tile's, 16 B apart, on every port core)
        self.zrw = ttnn.from_torch(
            torch.zeros(len(pc), 4 * self.RT, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(self.port_crs, (1, 4 * self.RT), ttnn.ShardOrientation.ROW_MAJOR),
            ),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        ttnn.synchronize_device(mesh)
        self.z = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.S, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        self.sched = TailSchedule(
            G=G, ring=self.ring, L=num_links, NTB=self.NTB, RT=self.RT, NBK=self.NBK, RP=self.RP, ties=self._ties()
        )
        # (a TP group is a mesh row: its chips by (mesh row, position))
        self.by_pos = {(c[0], self._pos(c)): c for c in self.chips}
        self.cache = {}

    def _pos(self, coord):
        return self.chips[coord]["rings"][0]["p"]

    def _ties(self):
        """{(p, q): direction} for the ring's far side (distance G / 2): the way the fabric routes it from p (as
        XGather3.ring_dir), the same on every mesh row (asserted)."""
        if not self.ring or self.G % 2:
            return {}
        node = lambda c: self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(*c))  # noqa: E731
        ties = {}
        for coord, ch in self.chips.items():
            p = self._pos(coord)
            q = (p + self.G // 2) % self.G
            qc = next(c for c in self.chips if c[0] == coord[0] and self._pos(c) == q)
            fd = ttnn._ttnn.fabric.get_eth_forwarding_direction(node(coord), node(ch["rings"][0]["next"]))
            d = "fwd" if ttnn._ttnn.fabric.get_eth_forwarding_direction(node(coord), node(qc)) == fd else "bwd"
            assert ties.setdefault((p, q), d) == d, "tail_a2a: ring ties differ between mesh rows"
        return ties

    # ---- the program ----
    def _program(self, coord, y, se, out):
        mesh, ch = self.mesh, self.chips[coord]
        p = self._pos(coord)
        virt = lambda c: mesh.worker_core_from_logical_core(c)  # noqa: E731
        node = lambda c: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*c))  # noqa: E731
        arr, rdy = (persist_l1.address(s) for s in self.sems)
        zr = int(self.zrw.buffer_address())
        prog = ttnn.ProgramDescriptor()
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        ports = ch["ports"]
        pr_rt, ps_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        sc = self.sched
        for (j, d, l), core in ports.items():
            ds = sc.dests(p, d)
            pr_rt[core.x][core.y] = [
                int(self.z.buffer_address()),
                self.NW,
                zr,
                self.RT,
                self.NTH,
                self.NTB,
                l,
                self.L,
                len(ds),
            ] + [q | part << 8 for q, part in ds]
            rs = sorted(sc.readies.get((p, d, l), set()))
            args = [int(out.buffer_address()), p, self.RT, self.NTB, l, self.L, arr, sc.expect.get((p, d, l), 0)]
            args += [rdy, len({q for q, _ in ds}), len(rs)]
            for xp, xd, xl in rs:
                xc = self.by_pos[(coord[0], xp)]
                v = virt(self.chips[xc]["ports"][(0, xd, xl)])
                n_ = node(xc)
                args += [v.x | v.y << 16, int(n_.chip_id), int(n_.mesh_id), sc.hops(p, xp, d)]
            args += [len(ds)]
            for q, part in ds:
                qc = self.by_pos[(coord[0], q)]
                v = virt(self.chips[qc]["ports"][(0, fag._OPP[d], l)])
                n_ = node(qc)
                args += [q | part << 8, int(n_.chip_id), int(n_.mesh_id), sc.hops(p, q, d)]
                args += [self.RT * sc.runs(part, l), v.x | v.y << 16]
            rg = ch["rings"][0]
            peer = rg["next"] if d == "fwd" else rg["prev"]
            if ds or rs:
                assert rg.get(f"{d}_links"), f"tail_a2a: {coord} has no {d} link"
                args += list(ttnn.setup_fabric_connection(node(coord), node(peer), rg[f"{d}_links"][l], prog, core))
            ps_rt[core.x][core.y] = args
        wr_rt, wc_rt, ww_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        # (the ports that carry something: the workers' z-ready words; a port with no destination neither waits nor
        # re-arms)
        pxy = [virt(c).x | virt(c).y << 16 for (j, d, l), c in ports.items() if sc.dests(p, d)]
        for i, core in enumerate(self.workers):
            c0 = i * self.W
            sa = int(se.buffer_address()) if se is not None else 0
            wr_rt[core.x][core.y] = [int(y.buffer_address()), sa, c0, self.RT, self.NTH]
            wc_rt[core.x][core.y] = [self.RT]
            ww_rt[core.x][core.y] = (
                [int(out.buffer_address()), int(self.z.buffer_address()), c0, self.RT, self.NTH]
                + [
                    self.NTB,
                    p,
                    zr,
                    len(pxy),
                ]
                + pxy
            )
        W, has_se = self.W, int(se is not None)
        fmt = lambda i, n: ttnn.CBDescriptor(  # noqa: E731
            total_size=n * self.page,
            core_ranges=self.work_crs,
            format_descriptors=[ttnn.CBFormatDescriptor(i, ttnn.bfloat16, self.page)],
        )
        # (every block W pages - the port's RP x GROUP: no block straddles a ring's end)
        cbs = [fmt(CB_RM, _ring_pages(2 * W, [W])), fmt(CB_T, _ring_pages(W, [W]))]
        cbs += [fmt(CB_SE, _ring_pages(2 * W, [W])), fmt(CB_OUT, _ring_pages(2 * W, [W]))]
        cbs.append(
            ttnn.CBDescriptor(
                total_size=_ring_pages(2 * self.group * self.RP, [self.group * self.RP]) * self.page,
                core_ranges=self.port_crs,
                format_descriptors=[ttnn.CBFormatDescriptor(CB_CH, ttnn.bfloat16, self.page)],
            )
        )
        prog.cbs = cbs
        st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(  # noqa: E731
            processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
        )
        comp = ttnn.ComputeConfigDescriptor()
        comp.math_fidelity = ttnn.MathFidelity.HiFi2
        comp.fp32_dest_acc_en = False
        comp.math_approx_mode = False
        pct = [CB_CH, self.page, self.RP, self.NBK, self.group, INC_EVERY]
        prog.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_WORKER_READER,
                source_type=st,
                core_ranges=self.work_crs,
                compile_time_args=[W, has_se] + acc(y) + (acc(se) if se is not None else acc(y)),
                runtime_args=wr_rt,
                config=dm("RISCV_1", ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_WORKER_COMPUTE,
                source_type=st,
                core_ranges=self.work_crs,
                compile_time_args=[W, has_se],
                runtime_args=wc_rt,
                config=comp,
            ),
            ttnn.KernelDescriptor(
                kernel_source=_WORKER_WRITER,
                source_type=st,
                core_ranges=self.work_crs,
                compile_time_args=[W] + acc(out) + acc(self.z),
                runtime_args=ww_rt,
                config=dm("RISCV_0", ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_PORT_READER,
                source_type=st,
                core_ranges=self.port_crs,
                compile_time_args=pct + acc(self.z),
                runtime_args=pr_rt,
                config=dm("RISCV_1", ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_PORT_SENDER,
                source_type=st,
                core_ranges=self.port_crs,
                compile_time_args=pct + acc(out),
                runtime_args=ps_rt,
                config=dm("RISCV_0", ttnn.NOC.NOC_1),
            ),
        ]
        return prog

    def run(self, y_rm, se, out=None):
        """y_rm [1, 1, S/N, H] ROW_MAJOR (DRAM), se [1, 1, S/N, H] TILE or None -> [1, 1, rows, H / TP] TILE."""
        if out is None:
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, self.rows, self.H // self.G]),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                self.mesh,
                ttnn.DRAM_MEMORY_CONFIG,
            )
        key = (int(y_rm.buffer_address()), int(se.buffer_address()) if se is not None else 0, int(out.buffer_address()))
        md = self.cache.get(key)
        if md is None:
            md = ttnn.MeshProgramDescriptor()
            for coord in self.chips:
                r, c = coord
                md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = self._program(
                    coord, y_rm, se, out
                )
            if len(self.cache) >= 16:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = md
        io = [y_rm, out, self.z, self.zrw] + ([se] if se is not None else [])
        if os.environ.get("MOE_TAIL_GENERIC") == "1":  # (A/B: the descriptor path, re-hashed and re-applied each call)
            ttnn.generic_op(io, md)
        else:
            # (prepared dispatch: md is kept per address set above, so each call patches nothing; its runtime args hold
            # only these tensors' base addresses, its CBs no tensor)
            prepared_generic_op(io, md)
        return out
