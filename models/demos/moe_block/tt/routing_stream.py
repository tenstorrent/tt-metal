# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""routing_stream - the fused head's routing exchange + combine planner (ttnn.bringup.combine_rf.routing_a2a, folded
into the head program): only the routing entries that point at a chip's experts cross chips, the planner builds
combine_plan's per-chip tables (counts, regions, token_index, table, plan) from local data.

Where it runs in the head:
  planner cores  the router cores (M = their number), on the BRISC once every router core's joint rows are written
                 (a barrier among them): routing_a2a's planner, its L1 a region of the head's arena, its exchange
                 semaphores global ones (re-armed), each destination's packet slices written into the L1 buffer of the
                 port that serves it.
  same-row ports the gather's movers (direction X+ / X-, link d % L): a polled stream of their sender (between their
                 other packets), once the M slices are in - routing_a2a's sender: one packet a (source, destination)
                 pair, empty ones too, multi-hop fabric unicast into the destination's DRAM slot (page parity N +
                 source), the last chunk a fused increment of its arrival counter (planner core 0).
  other rows     first hop Y (the fabric's route order: Y, then X): Y+ / Y- senders on the watcher core's BRISC.
  own chip       on the Y sender if there is one, else on a same-row port: a local write + increment.
"""

import os

import torch
from ttnn.bringup.combine_rf.routing_a2a import (
    _PLAN,
    PLAN_WORDS,
    _al16,
    header_bytes,
    routing_a2a_sizes,
    slot_bytes,
    worst_rows,
)

import ttnn
from ttnn.bringup.combine_rf import persist_l1
from ttnn.bringup.combine_rf.sparse_combine_l1 import tie_mode

DIR_OF = {"fwd": 0, "bwd": 1}  # (the gather's TP directions on X+ / X-: checked against the chips' coordinates)


def _plan_fn(subs):
    """routing_a2a's planner kernel as a function in namespace ra: ra::plan(a0, B) - its CB base B (an arena region)
    and every semaphore by address (args), CT accessors at RA_CT0."""
    src = _PLAN
    head, body = src.split("void kernel_main() {", 1)
    head = head.replace('#include <cstdint>\n#include "api/dataflow/dataflow_api.h"\n', "")
    head = head.replace("TensorAccessorArgs<0>()", "TensorAccessorArgs<__CT0__>()")
    body = body.replace("size_t a = 0;", "size_t a = a0;")
    body = body.replace("const uint32_t B = get_write_ptr(0);", "")
    body = body.replace(
        "const uint32_t go_sem = get_semaphore(0), parts_sem = get_semaphore(1), ex_sem = get_semaphore(2);",
        "const uint32_t go_sem = get_arg_val<uint32_t>(a++), parts_sem = get_arg_val<uint32_t>(a++), "
        "ex_sem = get_arg_val<uint32_t>(a++);",
    )
    # (planner instances need not share one L1 base or one set of semaphores - two run on a core, BRISC and NCRISC:
    # after the cores' NoC (x, y), every instance's base, exchange and go semaphores)
    body = body.replace(
        """        py[i] = get_arg_val<uint32_t>(a++);
    }
""",
        """        py[i] = get_arg_val<uint32_t>(a++);
    }
    uint32_t bb[M], exs[M], gos[M];
    for (uint32_t i = 0; i < M; ++i) bb[i] = get_arg_val<uint32_t>(a++);
    for (uint32_t i = 0; i < M; ++i) exs[i] = get_arg_val<uint32_t>(a++);
    for (uint32_t i = 0; i < M; ++i) gos[i] = get_arg_val<uint32_t>(a++);
""",
    )
    for a_, b_ in (
        ("get_noc_addr(px[r], py[r], B + O_EX + q * N * 4)", "get_noc_addr(px[r], py[r], bb[r] + O_EX + q * N * 4)"),
        (
            "noc_semaphore_inc(get_noc_addr(px[r], py[r], ex_sem), 1)",
            "noc_semaphore_inc(get_noc_addr(px[r], py[r], exs[r]), 1)",
        ),
        (
            "noc_semaphore_inc(get_noc_addr(px[r], py[r], go_sem), 1)",
            "noc_semaphore_inc(get_noc_addr(px[r], py[r], gos[r]), 1)",
        ),
        (
            "get_noc_addr(px[0], py[0], B + O_PLAN + (16 + aa) * 4)",
            "get_noc_addr(px[0], py[0], bb[0] + O_PLAN + (16 + aa) * 4)",
        ),
        ("get_noc_addr(px[0], py[0], B + O_PIN + q * 64)", "get_noc_addr(px[0], py[0], bb[0] + O_PIN + q * 64)"),
        (
            "get_noc_addr(px[r], py[r], B + O_PSUM + ((d / M) * M + q) * BB)",
            "get_noc_addr(px[r], py[r], bb[r] + O_PSUM + ((d / M) * M + q) * BB)",
        ),
    ):
        assert body.count(a_) == 1, a_
        body = body.replace(a_, b_)
    # (per sender core its own ready word: (x, y, ready) - the watcher core's two Y senders count apart)
    for a_, b_ in (
        ("    a += 2 * NPORT;\n", "    a += 3 * NPORT;\n"),
        (
            "            const uint32_t x = get_arg_val<uint32_t>(port_arg + 2 * i), y = get_arg_val<uint32_t>(port_arg + 2 * i + 1);\n"
            "            noc_semaphore_inc(get_noc_addr(x, y, ready_sem), 1);",
            "            const uint32_t x = get_arg_val<uint32_t>(port_arg + 3 * i), y = get_arg_val<uint32_t>(port_arg + 3 * i + 1);\n"
            "            noc_semaphore_inc(get_noc_addr(x, y, get_arg_val<uint32_t>(port_arg + 3 * i + 2)), 1);",
        ),
    ):
        assert body.count(a_) == 1, a_
        body = body.replace(a_, b_)
    # (the port buffers: absolute addresses, not offsets from the planner's base)
    a_ = "const uint32_t pb = B + get_arg_val<uint32_t>(dst_arg + 3 * d + 2);"
    assert body.count(a_) == 1, a_
    body = body.replace(a_, "const uint32_t pb = get_arg_val<uint32_t>(dst_arg + 3 * d + 2);")
    # (global semaphores: exact waits, then re-armed for the next call - every increment of this call has landed)
    body = body.replace(
        "noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ex_sem), M - 1);",
        "noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ex_sem), M - 1);\n"
        "    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ex_sem) = 0;",
    )
    body = body.replace(
        "noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_sem), 1);",
        "noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_sem), 1);\n"
        "        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_sem) = 0;",
    )
    body = body.replace(
        "noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(parts_sem), M - 1);",
        "noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(parts_sem), M - 1);\n"
        "        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(parts_sem) = 0;",
    )
    assert body.count("get_arg_val<uint32_t>(a++), parts_sem") == 1 and body.count("= 0;") >= 3
    # (zones: the packet wait, the counting / exchange before it)
    body = body.replace(
        "    // (5) every packet of the call: core 0 waits, then wakes the others\n    if (q == 0) {",
        '    // (5) every packet of the call: core 0 waits, then wakes the others\n    {\n    DeviceZoneScopedN("SE-RPWAIT");\n    if (q == 0) {',
    )
    body = body.replace("    ts[4] = get_timestamp_32b();", "    }\n    ts[4] = get_timestamp_32b();")
    body = body.replace(
        "const InterleavedAddrGen<true> dg = {.bank_base_address = dbg_addr, .page_size = 64};\n        noc_async_write(B + O_DBG, get_noc_addr(q, dg), 64);",
        "const InterleavedAddrGen<true> dg = {.bank_base_address = dbg_addr, .page_size = 128};\n        noc_async_write(B + O_DBG, get_noc_addr(q, dg), 128);",
    )
    # (RA_DBG: core 0 stamps each arrival - the debug page's words 16 .. 23 - instead of one blocking wait)
    body = body.replace(
        "noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ca), N);",
        "{ volatile tt_l1_ptr uint32_t* cw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ca); uint32_t seen = 0;\n"
        "          while (seen < N) { invalidate_l1_cache(); const uint32_t v = *cw; while (seen < v && seen < N) "
        "{ if (seen < 8) reinterpret_cast<uint32_t*>(B + O_DBG)[16 + seen] = get_timestamp_32b(); ++seen; } } }",
    )
    # (RA_DBG: my expert range's size and entries - the debug page's words 24, 25)
    for a_, b_ in (
        ("    uint32_t ts[10];\n", "    uint32_t ts[10], nent_dbg = 0;\n"),
        (
            "                const uint32_t w = eo[x];\n                ++et[",
            "                const uint32_t w = eo[x];\n                ++nent_dbg;\n                ++et[",
        ),
        (
            "        d[13] = B;\n",
            "        d[13] = B;\n        d[24] = e1 - e0;\n        d[25] = nent_dbg;\n        d[26] = tx_dbg[0];\n        d[27] = tx_dbg[1];\n        d[28] = tx_dbg[2];\n        d[29] = tx_dbg[3];\n",
        ),
        (
            "    uint32_t tot[EPC], reg[EPC], ent[N];\n",
            "    uint32_t tx_dbg[4];\n    tx_dbg[0] = get_timestamp_32b();\n    uint32_t tot[EPC], reg[EPC], ent[N];\n",
        ),
        (
            "        for (uint32_t i = 0; i < ER * S / 2; ++i) e32[i] = 0;\n    }\n",
            "        for (uint32_t i = 0; i < ER * S / 2; ++i) e32[i] = 0;\n    }\n    tx_dbg[1] = get_timestamp_32b();\n",
        ),
        ("    uint32_t off = 0, T = 0;\n", "    tx_dbg[2] = get_timestamp_32b();\n    uint32_t off = 0, T = 0;\n"),
        (
            "    const uint32_t ER = e1 - e0;\n",
            "    tx_dbg[3] = get_timestamp_32b();\n    const uint32_t ER = e1 - e0;\n",
        ),
    ):
        assert body.count(a_) == 1, a_
        body = body.replace(a_, b_)
    fn = "namespace ra {\n" + head + "FORCE_INLINE void plan(uint32_t a0, uint32_t B) {" + body + "}  // namespace ra\n"
    for k, v in subs.items():
        fn = fn.replace(f"__{k}__", str(int(v)))
    assert "__" not in fn.replace("__attribute__", ""), "unfilled planner placeholder"
    return fn


# a routing_a2a sender as a polled stream (a same-row port's sender, or the Y sender kernel): once the planner cores'
# M slices are in (ready >= M), the headers' totals, then one packet a destination (max-payload chunks, multi-hop
# unicast routed to its chip, the last a fused increment of its arrival counter); my own chip's (SELF): a local write
# + increment. args at r0: nd (0: no role), nr (remote), self, my buffer, PCAP, slot tensor address, c0, c1, planner
# core 0 (x, y), calls (global: this core's call count), ready (global), mesh, then the destinations' chip ids [nd]
RA_SEND = r"""
#ifndef RA_NF_
#define RA_NF_ 16  // (a forwarding port's slots: their flags, then the landed count - RoutingStream.NF)
#endif
// one packet's way on: to the next hop (kind 0: the destination's DRAM slot page par N + src, its arrival counter on
// planner core 0; kind 1: the next forwarding port's DRAM slot page (fbase: the forward slots tensor, slot | page << 8),
// its flag on that port), chunks of MAXP, the last with the increment. (A forward slot in DRAM, not the port's L1: a
// packet can land before that chip's head runs - in L1 the program still running there overwrote it, and the forwarder
// then sent a garbage-sized packet past its L1's end into the next hop's L1)
template <typename C, typename H>
FORCE_INLINE void ra_hop(C& conn, H hdr, uint32_t src_l1, uint32_t bytes, uint32_t a, uint32_t par, uint32_t srcid,
                         uint32_t PCAP, uint32_t sl, uint32_t c0, uint32_t c1, uint32_t px, uint32_t py, uint32_t mesh,
                         uint32_t fbase, uint32_t flags, uint32_t RA_N, uint32_t MAXP) {
    const uint32_t chip = get_arg_val<uint32_t>(a), kind = get_arg_val<uint32_t>(a + 1);
    const uint32_t nx = get_arg_val<uint32_t>(a + 2), ny = get_arg_val<uint32_t>(a + 3), slot = get_arg_val<uint32_t>(a + 4);
    uint64_t dst, inc;
    if (kind == 0) {
        const InterleavedAddrGen<true> slg = {.bank_base_address = sl, .page_size = PCAP};
        dst = get_noc_addr(par * RA_N + srcid, slg);
        inc = get_noc_addr(px, py, par ? c1 : c0);
    } else {
        const InterleavedAddrGen<true> fg = {.bank_base_address = fbase, .page_size = PCAP};
        dst = get_noc_addr(slot >> 8, fg);
        inc = get_noc_addr(nx, ny, flags + 16 * (slot & 0xFF));
    }
    for (uint32_t off = 0; off < bytes; off += MAXP) {
        const uint32_t sz = bytes - off < MAXP ? bytes - off : MAXP;
        noc_async_writes_flushed();  // (the header's previous packet has left)
        (void)fabric_set_unicast_route(hdr, static_cast<uint16_t>(chip), static_cast<uint16_t>(mesh));
        if (off + sz >= bytes) {
            hdr->to_noc_fused_unicast_write_atomic_inc(NocUnicastAtomicIncFusedCommandHeader{dst + off, inc, 1, true}, sz);
        } else {
            hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst + off}, sz);
        }
        conn.wait_for_empty_write_slot();
        conn.send_current_slot_non_blocking(src_l1 + off, sz, reinterpret_cast<uint32_t>(hdr));
    }
    if (kind == 1) {  // (+1 on the next forwarder's landed count, behind the packet: its poll reads one word)
        noc_async_writes_flushed();
        hdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(nx, ny, flags + 16 * RA_NF_), 1, true});
        conn.wait_for_empty_write_slot();
        conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
    }
}
FORCE_INLINE uint32_t ra_bytes(uint32_t pb, uint32_t RA_M, uint32_t RA_HBP) {
    const uint32_t* nq = reinterpret_cast<const uint32_t*>(pb);
    uint32_t n = 0;
    for (uint32_t q = 0; q < RA_M; ++q) n += nq[q];
    return (RA_HBP + 4 * n + 63) / 64 * 64;
}
struct RASend {
    uint32_t r0, nd, nr, self_, B, PCAP, sl, c0, c1, px, py, calls, ready, mesh, fbase, flags, done;
    void init(uint32_t a) {
        r0 = a; nd = get_arg_val<uint32_t>(a);
        if (!nd) { done = 1; return; }
        nr = get_arg_val<uint32_t>(a + 1); self_ = get_arg_val<uint32_t>(a + 2); B = get_arg_val<uint32_t>(a + 3);
        PCAP = get_arg_val<uint32_t>(a + 4); sl = get_arg_val<uint32_t>(a + 5); c0 = get_arg_val<uint32_t>(a + 6);
        c1 = get_arg_val<uint32_t>(a + 7); px = get_arg_val<uint32_t>(a + 8); py = get_arg_val<uint32_t>(a + 9);
        calls = get_arg_val<uint32_t>(a + 10); ready = get_arg_val<uint32_t>(a + 11); mesh = get_arg_val<uint32_t>(a + 12);
        fbase = get_arg_val<uint32_t>(a + 13); flags = get_arg_val<uint32_t>(a + 14);
        done = 0;
    }
    static uint32_t len(uint32_t a) { const uint32_t n = get_arg_val<uint32_t>(a); return n ? 15 + 5 * n : 1; }
    template <typename C, typename H>
    void poll(C& conn, H hdr, bool blocking, uint32_t RA_N, uint32_t RA_M, uint32_t RA_EPC, uint32_t RA_ME, uint32_t RA_HBP, uint32_t RA_BB, uint32_t RA_BBP, uint32_t MAXP) {
        if (done) return;
        volatile tt_l1_ptr uint32_t* rd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready);
        invalidate_l1_cache();
        if (*rd < RA_M) {
            if (!blocking) return;
            noc_semaphore_wait_min(rd, RA_M);
        }
        DeviceZoneScopedN("SE-RASEND");
        noc_semaphore_inc(get_noc_addr(ready), 0u - RA_M);
        noc_async_atomic_barrier();
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* cl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(calls);
        const uint32_t par = cl[0] & 1;  // (every port core and planner core counts its calls: they stay in step)
        cl[0] = cl[0] + 1;
        for (uint32_t j = 0; j < nd; ++j) {
            const uint32_t pb = B + j * PCAP;
            const uint32_t* nq = reinterpret_cast<const uint32_t*>(pb);
            // (the per-expert totals: the planners' - the destination's owner instance wrote them into the header)
            reinterpret_cast<uint32_t*>(pb)[14] = par;  // (for the forwarders: the call's parity, the source)
            reinterpret_cast<uint32_t*>(pb)[15] = RA_ME;
            (void)nq;
            const uint32_t bytes = ra_bytes(pb, RA_M, RA_HBP);
            if (j < nr) {
                ra_hop(conn, hdr, pb, bytes, r0 + 15 + 5 * j, par, RA_ME, PCAP, sl, c0, c1, px, py, mesh, fbase, flags, RA_N, MAXP);
            } else {  // (SELF: my own chip's packet)
                const InterleavedAddrGen<true> slg = {.bank_base_address = sl, .page_size = PCAP};
                noc_async_write(pb, get_noc_addr(par * RA_N + RA_ME, slg), bytes);
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(px, py, par ? c1 : c0), 1);
            }
        }
        noc_async_writes_flushed();
        done = 1;
    }
};
// a forwarding port's slots: each packet (of another chip, on its way) landed in its DRAM slot page with its flag set -
// read into the scratch slot (L1, this program's) and sent on to its next hop (any order), the flag re-armed. args at r0:
// n, scratch base, flags, PCAP, slot tensor, c0, c1, planner core 0 (x, y), mesh, the forward slots tensor, this port's
// first page, then per slot (next chip, kind, x, y, slot | page << 8)
struct RAFwd {
    uint32_t r0, n, fb, fl, PCAP, sl, c0, c1, px, py, mesh, fd, pg0, left, lastc;
    uint32_t sent[RA_NF_];
    void init(uint32_t a) {
        r0 = a; n = get_arg_val<uint32_t>(a); left = n; lastc = 0;
        if (!n) return;
        fb = get_arg_val<uint32_t>(a + 1); fl = get_arg_val<uint32_t>(a + 2); PCAP = get_arg_val<uint32_t>(a + 3);
        sl = get_arg_val<uint32_t>(a + 4); c0 = get_arg_val<uint32_t>(a + 5); c1 = get_arg_val<uint32_t>(a + 6);
        px = get_arg_val<uint32_t>(a + 7); py = get_arg_val<uint32_t>(a + 8); mesh = get_arg_val<uint32_t>(a + 9);
        fd = get_arg_val<uint32_t>(a + 10); pg0 = get_arg_val<uint32_t>(a + 11);
        for (uint32_t k = 0; k < RA_NF_; ++k) sent[k] = 0;
    }
    static uint32_t len(uint32_t a) { const uint32_t n = get_arg_val<uint32_t>(a); return n ? 12 + 5 * n : 1; }
    template <typename C, typename H>
    void poll(C& conn, H hdr, bool blocking, uint32_t RA_N, uint32_t RA_M, uint32_t RA_HBP, uint32_t MAXP) {
        // (the landed count - one increment a packet, behind its data and flag - says when a scan of the flags can find
        // something: a poll between the other streams' packets reads one word)
        volatile tt_l1_ptr uint32_t* ctr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fl + 16 * RA_NF_);
        while (left) {
            bool any = false;
            invalidate_l1_cache();
            const uint32_t c = *ctr;
            if (c == lastc) {
                if (!blocking) return;
                continue;
            }
            lastc = c;
            for (uint32_t k = 0; k < n; ++k) {
                volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fl + 16 * k);
                if (sent[k] || *f == 0) continue;
                DeviceZoneScopedN("SE-RAFWD");
                const uint32_t pb = fb + k * PCAP;
                {
                    const InterleavedAddrGen<true> fg = {.bank_base_address = fd, .page_size = PCAP};
                    noc_async_read(get_noc_addr(pg0 + k, fg), pb, PCAP);
                    noc_async_read_barrier();
                    invalidate_l1_cache();
                }
                const uint32_t par = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pb)[14];
                const uint32_t src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pb)[15];
                uint32_t bytes = ra_bytes(pb, RA_M, RA_HBP);
                if (bytes > PCAP) {  // (debug: a slot whose counts say more than a slot holds - not this call's
                                     // packet: recorded in the flag's spare words (bytes, how often), sent clamped)
                    f[1] = bytes;
                    f[2] = f[2] + 1;
                    f[3] = src;
                    bytes = PCAP;
                }
                ra_hop(conn, hdr, pb, bytes, r0 + 12 + 5 * k, par, src, PCAP, sl, c0, c1, px, py, mesh, fd, fl, RA_N, MAXP);
                // (re-arm: one packet a slot a call - by an atomic decrement, ordered with the previous hop's remote
                // increments: a plain store could wipe a next call's increment landing between the read and the store)
                noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(f)), 0u - 1u);
                sent[k] = 1;
                --left;
                any = true;
            }
            if (!blocking && !any) return;
        }
        if (blocking && n) {  // (re-arm once every packet's count has landed: by subtracting - a next call's packets may
                              // already land here (another row a call ahead: its count must survive, or the next call's
                              // poll never scans its flag - the forward lost, the destination's planner waiting forever)
            while (*ctr < n) invalidate_l1_cache();
            noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(ctr)), 0u - n);
            noc_async_atomic_barrier();
            lastc = 0;
        }
        noc_async_writes_flushed();
    }
};
"""

# the Y senders (watcher core BRISC): routing_a2a's sender alone, its own fabric connection toward the Y neighbour
_Y_SEND = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
#include "api/debug/kernel_profiler.hpp"
"""
    + RA_SEND
    + r"""
void kernel_main() {
    RASend rs;
    rs.init(0);
    size_t a = RASend::len(0);
    const uint32_t remote = get_arg_val<uint32_t>(a++);
    if (remote) {
        const uint32_t hello = get_arg_val<uint32_t>(a++), pc = get_arg_val<uint32_t>(a++), pm = get_arg_val<uint32_t>(a++);
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
        conn.open();
        // measurement only: "started" to the Y neighbour's watcher; the neighbour's landing here is stamped when seen
        // (between the sends' polls, never waited on before they are done) - the rows' launch skew
        (void)fabric_set_unicast_route(hdr, static_cast<uint16_t>(pc), static_cast<uint16_t>(pm));
        hdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(my_x[0], my_y[0], hello), 1, true});
        conn.wait_for_empty_write_slot();
        conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
        volatile tt_l1_ptr uint32_t* hw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hello);
        bool seen = false;
        while (!rs.done) {
            invalidate_l1_cache();
            if (!seen && *hw) {
                DeviceTimestampedData("SE-YHELLO", 1);
                seen = true;
            }
            rs.poll(conn, hdr, false, RA_N_, RA_M_, RA_EPC_, RA_ME_, RA_HBP_, RA_BB_, RA_BBP_, RA_MAXP_);
        }
        noc_semaphore_wait_min(hw, 1);
        if (!seen) DeviceTimestampedData("SE-YHELLO", 0);  // (late: after the sends)
        noc_semaphore_inc(get_noc_addr(hello), 0u - 1u);  // (re-arm)
        noc_async_atomic_barrier();
        rs.poll(conn, hdr, true, RA_N_, RA_M_, RA_EPC_, RA_ME_, RA_HBP_, RA_BB_, RA_BBP_, RA_MAXP_);
        conn.close();
    } else {
        struct NoConn { void wait_for_empty_write_slot() {} void send_current_slot_non_blocking(uint32_t, uint32_t, uint32_t) {} void send_payload_flush_blocking_from_address(uint32_t, uint32_t) {} } nc;
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = nullptr;
        rs.poll(nc, hdr, true, RA_N_, RA_M_, RA_EPC_, RA_ME_, RA_HBP_, RA_BB_, RA_BBP_, RA_MAXP_);
    }
    noc_async_atomic_barrier();
}
"""
)


class RoutingStream:
    """The routing exchange + planner in the head (module docstring). router: the head's RouterStream (its cores are the
    planner cores; its out tensor the own joint rows); xg: the gather (its ports); watcher: the watcher core."""

    def __init__(
        self, mesh_device, xg, router, *, S, K, EPC, A, L, luts, nps, rows_cap, ycores, arena, flags, fslots=16
    ):
        """ycores: the logical cores the Y senders run on (BRISC; one a needed Y direction); arena: (base, bytes free
        from offset off0 on the router cores: the planner's region; the ports' and Y cores' buffers in the arena's
        range there)."""
        self.mesh, self.xg, self.router = mesh_device, xg, router
        R, C = tuple(mesh_device.shape)
        self.R, self.C, self.N = R, C, R * C
        self.S, self.K, self.EPC, self.A, self.L = S, K, EPC, A, L
        self.luts, self.nps, self.rows_cap = luts, nps, rows_cap
        # planner instances: every router core's BRISC, then NCRISCs while there are experts (an instance needs one)
        # (at most 14: a packet header's words 14 / 15 carry the call's parity and the source (RASend) - words 0 .. M - 1
        # are the slices' counts; 15 or 16 instances (K3 at 1024 rows: 8 router cores) overwrote them - a hang)
        # (worst-case capacity: the plan tensors (DRAM) hold every row a chip can get, an instance's L1 every entry it
        # can get - packed routing never drops a row. An instance's L1 grows as M falls (its share of T): the most
        # instances whose regions fit the router cores - one a core under the router's overlay or past its buffers,
        # or two (the NCRISCs too) - K3 at 1024 rows on the 8x4: 8 of 14, its worst case did not fit twice a core)
        al = lambda v: (v + 63) // 64 * 64  # noqa: E731
        base, abytes, off0 = arena
        nc = len(router.cores)
        assert self.N * EPC <= 1024 and S <= 2048 and EPC <= 64 and S % 32 == 0
        for M in range(min(2 * nc, EPC, A, 14), 0, -1):
            sz = routing_a2a_sizes(N=self.N, S=S, K=K, EPC=EPC, A=A, rows_cap=rows_cap, M=M)
            step, per = al(sz["plan_l1"]), -(-M // nc)
            pl0 = 0 if per * step <= router.off["bias"] else al(router.bytes)
            if sz["fits"] and pl0 + per * step <= abytes:
                break
        assert sz["fits"], sz["why"]
        assert pl0 + per * step <= abytes, (
            f"router cores: no room for the planner regions ({step} B each, {per} a core past {pl0} B; the arena "
            f"{abytes} B)"
        )
        self.M = M
        assert rows_cap >= worst_rows(self.N, S, K, EPC), f"rows_cap {rows_cap} < {worst_rows(self.N, S, K, EPC)}"
        self.inst_cap = sz["inst_cap"]
        self.plan_l1 = sz["plan_l1"]
        self.HBP = header_bytes(EPC, M)
        self.BB, self.BBP = _al16(2 * EPC), _al16(2 * (EPC + 1))
        self.PCAP = slot_bytes(S, K, EPC, M)
        self.MAXP = int(ttnn.get_tt_fabric_max_payload_size_bytes()) // 64 * 64
        self.ycores = ycores
        ycrs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in ycores])
        self.yhello = ttnn.from_torch(  # (the Y neighbour's "started" lands here: a zero word, re-armed each call)
            torch.zeros(len(ycores), 16, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                persist_l1.buffer_type(),
                ttnn.ShardSpec(ycrs, (1, 16), ttnn.ShardOrientation.ROW_MAJOR),
            ),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.wrap_x = bool(xg.ring)
        self.wrap_y = False  # (the LoudBox's SP axis: a line of 2; the Galaxy's SP ring: torus Y)
        cfg = ttnn.get_fabric_config()
        if cfg in (ttnn.FabricConfig.FABRIC_2D_TORUS_XY, ttnn.FabricConfig.FABRIC_2D_TORUS_Y) and R > 2:
            self.wrap_y = True
        # the planner instances' regions overlay the router's matmul / partial / zone buffers (dead once its top-K ran:
        # the planner starts after the joint rows), or (where they don't fit under the bias - K3 E 896 at 1280 rows on a
        # 4x4) go past the router's buffers, in the rest of the arena's range on the router cores: nothing else there
        self.pl_step = step
        self.pbases = [router.base + pl0 + (q // nc) * self.pl_step for q in range(M)]
        self.pbase = self.pbases[0]
        # the senders' packet buffers (the arena's range on their cores): the gather's ports after their logits buffer,
        # each Y sender on the watcher core from the arena's start (the second Y direction after the first); the
        # forwarding slots after a port's own packets
        al = lambda v: (v + 63) // 64 * 64  # noqa: E731
        self.xbase = base + al(router.pbuf_sub)
        self.ybase = base
        nx = max(len(v) for c in xg.chips for k, v in self.serve(c).items() if k[0] == "x")
        ny = max([len(v) for c in xg.chips for k, v in self.serve(c).items() if k[0] == "y"] + [0])
        self.fbase = self.xbase + nx * self.PCAP
        self.flags = flags  # (the forwarding slots' flags: 16 B a slot, zeroed L1 - fslots of them, then the count)
        self.NF = fslots
        nf = max([len(v) for v in self.fwd_plan()[0].values()] + [0])
        assert al(router.pbuf_sub) + (nx + nf) * self.PCAP <= abytes, (
            f"ports: no L1 for the routing buffers ({al(router.pbuf_sub)} B logits + {nx} + {nf} slots x {self.PCAP} B = "
            f"{al(router.pbuf_sub) + (nx + nf) * self.PCAP} B > the arena's {abytes} B)"
        )
        assert 2 * ny * self.PCAP <= abytes, "the watcher core: no L1 for the Y senders' buffers"
        self.ny = ny
        from loguru import logger as _lg

        _lg.info(
            f"RoutingStream: PCAP {self.PCAP} MAXP {self.MAXP} HBP {self.HBP} arena base {base:#x} bytes {abytes} xbase "
            f"{self.xbase:#x} nx {nx} fbase {self.fbase:#x} nf {nf} (slots end {self.fbase + nf * self.PCAP:#x}) "
            f"flags {self.flags:#x}"
        )

        sh = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(0, 1))

        def dev(t):
            return ttnn.from_torch(
                t,
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=sh,
            )

        z = lambda *s: torch.zeros(R, C, *s, dtype=torch.int32)  # noqa: E731
        NG = self.N * EPC
        self.outs = dict(
            counts=dev(z(1, NG)),
            regions=dev(z(1, NG)),
            token_index=dev(z(1, rows_cap)),
            table=dev(z(rows_cap // 32, 64)),
            plan=dev(z(1, PLAN_WORDS)),
        )
        self.slots = dev(z(2 * self.N, self.PCAP // 4))
        # the forwarding ports' slots: a DRAM page each (PCAP), per forwarding port key 16 - the same on every chip
        self.fkeys = sorted({k for (_, k) in self.fwd_plan()[0]})
        self.fslots = dev(z(max(1, len(self.fkeys)) * self.NF, self.PCAP // 4))
        import os

        self.dbg = dev(z(16, 32)) if os.environ.get("RA_DBG") else None  # (the planner's phase stamps, a page a core)
        pc = router.crs
        c0 = ttnn.CoreRangeSet([ttnn.CoreRange(router.cores[0], router.cores[0])])
        ports = sorted({(c.x, c.y) for ch in xg.chips.values() for c in ch["ports"].values()})
        senders = ports + [(c.x, c.y) for c in ycores]
        scs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in senders])
        gs = lambda crs: persist_l1.semaphore(mesh_device, crs, 0)  # noqa: E731
        self.sems = dict(
            c0=gs(c0),
            c1=gs(c0),
            calls_p=gs(pc),
            calls_p2=gs(pc),
            go=gs(pc),
            go2=gs(pc),
            parts=gs(pc),
            ex=gs(pc),
            ex2=gs(pc),
            rows=gs(pc),
            rows2=gs(pc),
            calls_s=gs(scs),
            calls_s2=gs(scs),  # (the second Y direction's sender: the watcher core's other RISC, its own call count)
            ready=gs(scs),
            ready2=gs(scs),  # (the second Y direction's sender's: the planners count each sender apart)
        )
        self.addr = {k: persist_l1.address(v) for k, v in self.sems.items()}

    # ---- the plan of one chip ----
    def _first_hop(self, s, d):
        R, C = self.R, self.C
        sr, sc, dr, dc = s // C, s % C, d // C, d % C
        if sr != dr:
            if self.wrap_y:
                return 2 if 2 * ((dr - sr) % R) < R else 3
            return 2 if dr > sr else 3
        if self.wrap_x:
            return 0 if 2 * ((dc - sc) % C) <= C else 1
        return 0 if dc > sc else 1

    def serve(self, coord):
        """{sender key: [destinations]} of chip coord: ("x", tp direction, link) the gather's ports, ("y", 2 | 3) the Y
        senders; my own chip last on the Y sender if any, else on the first port."""
        me = coord[0] * self.C + coord[1]
        out = {}
        for dd in range(self.N):
            if dd == me:
                continue
            out.setdefault(self.path(me, dd)[0][1], []).append(dd)
        ykeys = sorted(k for k in out if k[0] == "y")
        own = ykeys[0] if ykeys else sorted(out)[0]
        out.setdefault(own, []).append(me)
        assert len(ykeys) <= 2, f"chip {coord}: {len(ykeys)} Y directions"
        return out

    def _xdir(self, coord):
        """{0 (X+) | 1 (X-): the gather's direction name toward it} at chip coord."""
        rg = self.xg.chips[coord]["rings"][0]
        out = {}
        for d in ("fwd", "bwd"):
            peer = rg["next"] if d == "fwd" else rg["prev"]
            if peer is not None and rg.get(f"{d}_links"):
                # (a line by the sign: modulo C 2 both neighbours are + 1)
                fwd = (peer[1] - coord[1]) % self.C == 1 if self.wrap_x else peer[1] > coord[1]
                out[0 if fwd else 1] = d
        return out

    def path(self, s, d):
        """The hops of the packet from chip s to chip d: [(chip, sender key)] - the first at s (its sender), the others
        forwarding ports (store-and-forward: every hop a local injection, nothing in transit through a busy row);
        Y first (the idle SP links, transit allowed there), then X along the destination row one chip a hop."""
        C = self.C
        sr, sc, dr, dc = s // C, s % C, d // C, d % C
        steps = []
        cur = (sr, sc)
        if sr != dr:
            steps.append((s, ("y", self._first_hop(s, d))))
            cur = (dr, sc)
        while cur[1] != dc:
            ci = cur[0] * C + cur[1]
            h = self._first_hop(ci, d)
            xd = self._xdir(cur)
            assert h in xd, f"chip {cur}: no X port toward {d}"
            l = d % self.L if (xd[h], d % self.L) in self._ports(cur) else 0
            steps.append((ci, ("x", xd[h], l)))
            cur = (cur[0], (cur[1] + (1 if h == 0 else -1)) % C)
        return steps

    def fwd_plan(self):
        """{(chip, key): [(src, dst, hop index)]} - the forwarding ports' slots (every path's hops after its first), in a
        fixed order; and per (src, dst) the slot index at each forwarding hop."""
        if getattr(self, "_fwd", None) is None:
            fw, at = {}, {}
            for s_ in range(self.N):
                for d in range(self.N):
                    if d == s_:
                        continue
                    for i, step in enumerate(self.path(s_, d)[1:], 1):
                        lst = fw.setdefault(step, [])
                        at[(s_, d, i)] = len(lst)
                        lst.append((s_, d, i))
            self._fwd = (fw, at)
        return self._fwd

    def next_hop(self, s, d, i):
        """Where hop i of the (s, d) packet goes: (next chip, kind 0 final | 1 forward, NoC x, NoC y, slot)."""
        steps = self.path(s, d)
        node = lambda c: self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(c // self.C, c % self.C))  # noqa: E731
        virt = self.mesh.worker_core_from_logical_core
        if i + 1 < len(steps):
            ci, key = steps[i + 1]
            core = virt(self._core_of((ci // self.C, ci % self.C), key))
            slot = self.fwd_plan()[1][(s, d, i + 1)]
            return [int(node(ci).chip_id), 1, core.x, core.y, slot | (self.fkeys.index(key) * self.NF + slot) << 8]
        return [int(node(d).chip_id), 0, 0, 0, 0]

    def fwd_block(self, coord, key):
        """A forwarding port's RAFwd args: [n, buffer base, flag base] + per slot its next hop (5 words)."""
        lst = self.fwd_plan()[0].get((coord[0] * self.C + coord[1], key), [])
        if not lst:
            return [0]
        virt = self.mesh.worker_core_from_logical_core
        p0 = virt(self.router.cores[0])
        node = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
        assert len(lst) <= self.NF, f"routing: a port forwards {len(lst)} packets > its {self.NF} slots"
        out = [len(lst), self.fbase, self.flags, self.PCAP, int(self.slots.buffer_address()), self.addr["c0"]]
        out += [self.addr["c1"], p0.x, p0.y, int(node.mesh_id)]
        out += [int(self.fslots.buffer_address()), self.fkeys.index(key) * self.NF]
        for s_, d, i in lst:
            out += self.next_hop(s_, d, i)
        return out

    def _ports(self, coord):
        return {(d, l): c for (_, d, l), c in self.xg.chips[coord]["ports"].items()}

    def _buf(self, coord, key):
        """A sender's packet buffer (its destination j at + j PCAP)."""
        if key[0] == "x":
            return self.xbase
        ykeys = sorted(k for k in self.serve(coord) if k[0] == "y")
        return self.ybase + ykeys.index(key) * self.ny * self.PCAP

    def _core_of(self, coord, key):
        if key[0] == "x":
            return self._ports(coord)[(key[1], key[2])]
        return self.ycores[0]  # (the watcher core: Y senders on its BRISC and, after the watch, its NCRISC)

    def planner_args(self, coord):
        """The planner's runtime args (routing_a2a's head + the planner core q appended by the caller per core)."""
        self.serve_cache = getattr(self, "serve_cache", {})
        self.serve_cache[coord] = sv = self.serve(coord)
        virt = self.mesh.worker_core_from_logical_core
        where = {}
        for key, dests in sv.items():
            core, b = self._core_of(coord, key), self._buf(coord, key)
            for j, d in enumerate(dests):
                where[d] = (core, j, b)
        o = self.outs
        head = [
            int(self.router_out.buffer_address()),
            int(self.slots.buffer_address()),
            self.addr["c0"],
            self.addr["c1"],
            None,  # (the instance's call count: per RISC)
            int(o["counts"].buffer_address()),
            int(o["regions"].buffer_address()),
            int(o["token_index"].buffer_address()),
            int(o["table"].buffer_address()),
            int(o["plan"].buffer_address()),
        ] + list(self.luts[coord])
        nc = len(self.router.cores)
        pv = [virt(self.router.cores[q % nc]) for q in range(self.M)]
        pxy = [w for v in pv for w in (v.x, v.y)]
        per = lambda k1, k2: [self.addr[k1] if q < nc else self.addr[k2] for q in range(self.M)]  # noqa: E731
        inst = self.pbases + per("ex", "ex2") + per("go", "go2")
        dst = []
        for d in range(self.N):
            core, j, b = where[d]
            v = virt(core)
            dst += [v.x, v.y, b + j * self.PCAP]
        senders = sorted(sv)
        port_args = [
            w
            for k in senders
            for v in [virt(self._core_of(coord, k))]
            for w in (v.x, v.y, self.addr[self._sem(sv, k)[1]])
        ]
        tail = [int(self.dbg.buffer_address()) if self.dbg is not None else 0, self.addr["ready"]]

        def args(q):
            h = list(head)
            h[4] = self.addr["calls_p"] if q < nc else self.addr["calls_p2"]
            sems = [self.addr["go"] if q < nc else self.addr["go2"], self.addr["parts"]]
            sems += [self.addr["ex"] if q < nc else self.addr["ex2"]]
            return h + [q, 0, 0] + pxy + inst + dst + port_args + tail + sems

        return args

    def kernel_subs(self, coord, ct0):
        return dict(
            S=self.S,
            K=self.K,
            EPC=self.EPC,
            R=self.R,
            C=self.C,
            ME=coord[0] * self.C + coord[1],
            A=self.A,
            L=self.L,
            P=self.nps[coord],
            TPP=32,
            M=self.M,
            HBP=self.HBP,
            BB=self.BB,
            BBP=self.BBP,
            PCAP=self.PCAP,
            NPORT=len(self.serve_cache[coord]),
            ROWS_CAP=self.rows_cap,
            ENT_CAP=self.inst_cap,
            WX=int(self.wrap_x),
            WY=int(self.wrap_y),
            YT=tie_mode(),  # (the combine's Y tie split: the plan's ports, as routing_a2a)
            DCACHE=1,
            PROBE_NOSTORE=0,
            CT0=ct0,
        )

    @staticmethod
    def _sem(sv, key):
        """A sender's (calls, ready) semaphores: the second Y direction's sender (the watcher core's other RISC, on the
        Galaxy's Y ring) its own pair - one shared call count would hand it the next call's parity."""
        second_y = key[0] == "y" and key != min(k for k in sv if k[0] == "y")
        return ("calls_s2", "ready2") if second_y else ("calls_s", "ready")

    def send_block(self, coord, key):
        """A sender's RASend args (key None or not serving: [0])."""
        sv = self.serve_cache.get(coord) or self.serve(coord)
        if key not in sv:
            return [0]
        dests = sv[key]
        me = coord[0] * self.C + coord[1]
        nr = len(dests) - int(dests[-1] == me)
        node = lambda i: self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(i // self.C, i % self.C))  # noqa: E731
        virt = self.mesh.worker_core_from_logical_core
        p0 = virt(self.router.cores[0])
        return [
            len(dests),
            nr,
            int(dests[-1] == me),
            self._buf(coord, key),
            self.PCAP,
            int(self.slots.buffer_address()),
            self.addr["c0"],
            self.addr["c1"],
            p0.x,
            p0.y,
            self.addr[self._sem(sv, key)[0]],
            self.addr[self._sem(sv, key)[1]],
            int(node(me).mesh_id),
            int(self.fslots.buffer_address()),  # (ra_hop's kind 1: the next forwarder's DRAM slot page)
            self.flags,
        ] + [w for d in dests for w in (self.next_hop(me, d, 0) if d != me else [0, 0, 0, 0, 0])]

    def defines(self, coord):
        me = coord[0] * self.C + coord[1]
        return "".join(
            f"#define {k} {v}\n"
            for k, v in dict(
                RA_NC_=len(self.router.cores),
                RA_NP_=self.router.RB,  # (the joint rows' makers: a row tile's - its half 0 with the split)
                RA_N_=self.N,
                RA_M_=self.M,
                RA_EPC_=self.EPC,
                RA_ME_=me,
                RA_HBP_=self.HBP,
                RA_BB_=self.BB,
                RA_BBP_=self.BBP,
                RA_MAXP_=self.MAXP,
                RA_NF_=self.NF,
            ).items()
        )

    def _y_args(self, coord, key, prog, core):
        node = lambda c: self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(*c))  # noqa: E731
        sv = self.serve_cache[coord]
        nr = len(sv[key]) - int(sv[key][-1] == coord[0] * self.C + coord[1])
        args = self.send_block(coord, key) + [int(nr > 0)]
        if nr:
            dy = 1 if key[1] == 2 else -1
            peer = ((coord[0] + dy) % self.R, coord[1])
            args += [int(self.yhello.buffer_address()), int(node(peer).chip_id), int(node(peer).mesh_id)]
            link = ttnn.get_forwarding_link_indices(node(coord), node(peer))[0]
            args += list(ttnn.setup_fabric_connection(node(coord), node(peer), link, prog, core))
        return args

    def watcher_tail(self, coord, prog):
        """The second Y direction's sender (Galaxy's Y ring), run by the watcher's NCRISC after its watch: (defines,
        args) to append to the watcher's, or ("", [])."""
        ykeys = sorted(k for k in self.serve_cache[coord] if k[0] == "y")
        if len(ykeys) < 2:
            return "", []
        return self.defines(coord), self._y_args(coord, ykeys[1], prog, self.ycores[0])

    def y_kernels(self, coord, prog):
        """The first Y direction's sender of chip coord (a watcher-core BRISC kernel), with its fabric connection."""
        sv = self.serve_cache[coord]
        ks = []
        for key in sorted(k for k in sv if k[0] == "y")[:1]:
            core = self._core_of(coord, key)
            args = self._y_args(coord, key, prog, core)
            rt = ttnn.RuntimeArgs()
            rt[core.x][core.y] = args
            ks.append(
                ttnn.KernelDescriptor(
                    kernel_source=self.defines(coord) + _Y_SEND,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
                    compile_time_args=[],
                    runtime_args=rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                    ),
                )
            )
        return ks
