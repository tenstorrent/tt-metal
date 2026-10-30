# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""x_gather3 - the MoE head's x all-gather over TP, ROW-CHUNK-MAJOR, straight into se_engine3's x-own L1.

Every chip p of a mesh row holds x [rows, kl] (TILE, DRAM). Chunk by chunk (CR row tiles), store-and-forward over
fabric unicasts (tp_gather's ports, line / ring schedule and call fence): a port per (direction, link) sends the
chunk's rows of its own shard (read from DRAM x), then each of the chunk's shards the upstream port landed here (read
from the owners' L1). A packet: up to RP tiles of one row tile within one K step, written at the step's owner core
(the engine's x families' owner tables) and slot. Row tile R (chunk-local lr) goes over link (lr / 2) % links. After each chunk a port
increments the downstream chip's watcher ((direction, link) counter) after each shard; the watcher raises XREADY on
every engine core per shard in the engine's order (own shard first - the engine loads it itself - then by arrival):
XREADY = m (G - 1) + k once chunk m's shard of rank k is in, so gate / up starts on the own K steps at once. x-own is
never reused within a call, so no credits: the call fence only."""

from ttnn.bringup.combine_rf import persist_l1
from ttnn.bringup.combine_rf.fabric_all_gather_py import _OPP, NOC0, NOC1, plan

import ttnn
from models.demos.moe_block.tt.router_stream import LSTREAM

# (the port cores' roles: the sender (Ethernet writes) on the NCRISC / NOC0, the reader on the BRISC / NOC1. The ports
# sit in one column: NOC1 (north, then west) funnels all of them through the same Ethernet-row links (~75 GB/s for all
# four, the gather 138 us vs a 69 us floor); NOC0 (east, then south) takes each port's own row toward its own Ethernet
# column (82 us). Head K3 ring of 4 452-459 -> 439-443 us, GLM 2x4 1084 -> 967. SE_PORT_NOC0=0: the old roles (A/B))
PORT_NOC0 = __import__("os").environ.get("SE_PORT_NOC0", "1") == "1"
from models.demos.moe_block.tt.routing_stream import RA_SEND
from models.demos.moe_block.tt.se_engine3 import _probes

INC_EVERY = 8
RSX = 10  # (the RS block's words past the entry order - the kernels' RSX)
CB = 0
DIRS_I = {"fwd": 0, "bwd": 1}

_CT = r"""
constexpr uint32_t KB = get_compile_time_arg_val(0), MT = get_compile_time_arg_val(1), SPS = get_compile_time_arg_val(2);
constexpr uint32_t RP = get_compile_time_arg_val(3), CR = get_compile_time_arg_val(4), NCH = get_compile_time_arg_val(5);
constexpr uint32_t NV = get_compile_time_arg_val(6), KLT = get_compile_time_arg_val(7), GROUP = get_compile_time_arg_val(8);
constexpr uint32_t INC = get_compile_time_arg_val(9), GX = get_compile_time_arg_val(10), GY = get_compile_time_arg_val(11);
constexpr uint32_t S0 = get_compile_time_arg_val(12), S1 = get_compile_time_arg_val(13), G = get_compile_time_arg_val(14);
constexpr uint32_t RSX = 10;  // (the RS block's words past the entry order: FINB, mailbox, zone, counter, P, FINB-free,
                              // zone credit, zone free, the zone-free signals a chunk, the credits a call expected)
"""

_WALK = r"""
constexpr uint32_t PPS = (KB + RP - 1) / RP, RPR = SPS * PPS;  // (packets a K step, a row tile)
// chunk m of shard q, link v: row tile R of the chunk (link (R / 2) % NV; part 1 / 2: even / odd R), step j, first
// tile kk0 within the step, n tiles; idx: the run's index in the link's full (part 0) walk of the chunk's shard
template <typename F>
FORCE_INLINE void walk(uint32_t m, uint32_t q, uint32_t part, uint32_t v, F&& f) {
    uint32_t cnt = 0;
    for (uint32_t R = m * CR; R < (m + 1) * CR; ++R) {
        const uint32_t lr = R - m * CR;  // (link and ring half by the chunk-local row: the host counts packets so)
        if ((lr / 2) % NV != v) continue;
        if (part == 0 || (lr & 1) == part - 1)
            for (uint32_t st = 0; st < SPS; ++st)
                for (uint32_t kk0 = 0, i = 0; kk0 < KB; kk0 += RP, ++i)
                    f(R, q * SPS + st, st * KB + kk0, kk0, KB - kk0 < RP ? KB - kk0 : RP, cnt * RPR + st * PPS + i);
        ++cnt;
    }
}
// x step j of chunk m, row tile R on a chip: its processing position there (the chip's shard rank, then the step
// within the shard), the position's owner core (NoC) and x-own address (se_engine3's layout). args at T: the chip's
// shard ranks [G]; fam0[S0] cols, fam1[S1] cols; per column: nown, family-1 base slot; column NoC x [GX]; row y [GY]
struct Dst { uint32_t x, y, addr; };
FORCE_INLINE Dst dst_of(uint32_t T, uint32_t xown, uint32_t m, uint32_t R, uint32_t j, uint32_t kk0) {
    const uint32_t pos = get_arg_val<uint32_t>(T + j / SPS) * SPS + j % SPS;
    T += G;
    const uint32_t f = pos & 1, h = pos / 2;
    const uint32_t col = f ? get_arg_val<uint32_t>(T + S0 + h % S1) : get_arg_val<uint32_t>(T + h % S0);
    const uint32_t t = T + S0 + S1;
    const uint32_t nown = get_arg_val<uint32_t>(t + 2 * col), b1 = get_arg_val<uint32_t>(t + 2 * col + 1);
    const uint32_t slot = f ? b1 + h / S1 : h / S0;
    const uint32_t lr = R - m * CR;
    const uint32_t addr = xown + (((m * nown + slot) * MT + lr % MT) * KB + kk0) * XTB;  // (XTB: x's tile bytes)
    return {get_arg_val<uint32_t>(t + 2 * GX + col), get_arg_val<uint32_t>(t + 3 * GX + lr / MT), addr};
}
"""

_RS_READER = r"""
// (down + reduce-scatter) gather my units: per unit (entry e, block b, halves), per core of my link in the unit's
// halves holding tiles of b: wait its hand-off on my mailbox word of it, NoC-read its bf16 sub-block from its FINB
// (compact in the chip's entry order) in chunks of RP tiles; a batch ends at a multiple of GROUP chunks, at the end
// of a unit and before a blocking wait (matmul_reduce_scatter's port reader). At the end my mailbox words re-armed.
// args at r0: nu, (e | b << 8 | mask << 16, slot0 | slot1 << 16) x nu, nc, (x | y << 8 | half << 16 | k << 20,
// counts (8 bits a block)) x nc, the chip's entry order [G], FINB address, mailbox address, .., FINB-free address.
// Chunk m of NCH: the mailbox targets count up (seen, the caller's, over the chunks); after chunk m (not the last) +1
// on the FINB-free word of every core read - its FINB is free for chunk m + 1; after the last the words re-armed
FORCE_INLINE void rs_gather(uint32_t r0, uint32_t& pos, uint32_t m, uint32_t* seen) {
    const uint32_t nu = get_arg_val<uint32_t>(r0), u0 = r0 + 1, nc = get_arg_val<uint32_t>(u0 + 2 * nu);
    if (nu == 0) return;
    const uint32_t c0 = u0 + 2 * nu + 1, en0 = c0 + 2 * nc;
    const uint32_t finb = get_arg_val<uint32_t>(en0 + G), mbox = get_arg_val<uint32_t>(en0 + G + 1);
    if (m == 0)
        for (uint32_t i = 0; i < nc; ++i) seen[i] = 0;
    uint8_t rdc[128];
    for (uint32_t i = 0; i < nc; ++i) rdc[i] = 0;
    uint32_t c = 0, need = 0, wptr = 0;  // (pos: chunks pushed since the phases began - the CB's batch alignment)
#ifdef SE_PROBE_RSSTAT
    DeviceZoneScopedN("SE-RSG");
    auto clk = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    uint32_t wprod = 0, wcb = 0, nw = 0;  // (cycles blocked on a producer's hand-off; on the CB's room)
#endif
    auto flush = [&]() {
        noc_async_read_barrier();
        cb_push_back(0, c * RP);
        pos += c;
        c = 0;
    };
    for (uint32_t un = 0; un < nu; ++un) {
        const uint32_t w0 = get_arg_val<uint32_t>(u0 + 2 * un), e = w0 & 0xFF, b = (w0 >> 8) & 0xFF, mask = (w0 >> 16) & 3;
        for (uint32_t i = 0; i < nc; ++i) {
            const uint32_t cw = get_arg_val<uint32_t>(c0 + 2 * i), cnts = get_arg_val<uint32_t>(c0 + 2 * i + 1);
            const uint32_t cnt = (cnts >> (8 * b)) & 0xFF;
            if (!((mask >> ((cw >> 16) & 1)) & 1) || cnt == 0) continue;
            uint32_t fo = 0;  // (the sub-block's FINB offset: the core's tiles of the entries before e)
            for (uint32_t e2 = 0; e2 < e; ++e2) fo += (cnts >> (8 * get_arg_val<uint32_t>(en0 + e2))) & 0xFF;
            volatile tt_l1_ptr uint32_t* mb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mbox + XG_HS * (cw >> 20));
            const uint32_t target = ++seen[i];
            invalidate_l1_cache();
            HWW(16, 0xA1000000u | un << 8 | i); HWW(17, target); HWW(18, *mb); HWW(19, mbox + XG_HS * (cw >> 20));
            if (*mb < target && c > 0) flush();  // (never hold reads while blocked on a producer)
#ifdef SE_PROBE_RSSTAT
            if (*mb < target) {
                const uint32_t t = clk();
                noc_semaphore_wait_min(mb, target);
                wprod += clk() - t;
                ++nw;
            }
#endif
            noc_semaphore_wait_min(mb, target);
            rdc[i] = 1;
            const uint64_t src = get_noc_addr(cw & 0xFF, (cw >> 8) & 0xFF, finb + fo * 2048);
            for (uint32_t t0 = 0; t0 < cnt; t0 += RP) {
                if (c == 0) {
                    need = GROUP - pos % GROUP;
#ifdef SE_PROBE_RSSTAT
                    const uint32_t t = clk();
                    cb_reserve_back(0, need * RP);
                    wcb += clk() - t;
#endif
                    cb_reserve_back(0, need * RP);
                    wptr = get_write_ptr(0);
                }
                const uint32_t n = cnt - t0 < RP ? cnt - t0 : RP;
                noc_async_read(src + t0 * 2048, wptr + c * RP * 2048, n * 2048);
                if (++c == need) flush();
            }
        }
        if (c > 0) flush();
    }
#ifdef SE_PROBE_RSSTAT
    DeviceTimestampedData("SE-RSG-WPROD", wprod);
    DeviceTimestampedData("SE-RSG-WCB", wcb);
    DeviceTimestampedData("SE-RSG-NW", nw);
#endif
    if (m + 1 < NCH) {  // (every read of the chunk landed - each unit ends in a flush: FINB free)
        const uint32_t ffa = get_arg_val<uint32_t>(en0 + G + 5);
        for (uint32_t i = 0; i < nc; ++i)
            if (rdc[i]) {
                const uint32_t cw = get_arg_val<uint32_t>(c0 + 2 * i);
                noc_semaphore_inc(get_noc_addr(cw & 0xFF, (cw >> 8) & 0xFF, ffa), 1);
            }
        noc_async_atomic_barrier();
        return;
    }
    for (uint32_t i = 0; i < nc; ++i)  // (re-arm: every hand-off of the call was waited for)
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mbox + XG_HS * (get_arg_val<uint32_t>(c0 + 2 * i) >> 20)) = 0;
}
"""

_LA_READER = r"""
// (K3's latent all-to-all) per destination block b of my direction, per core of my link holding rows of b: wait its
// hand-off on my latent word of it, NoC-read its rows [r0, r0 + nr) of its latent block [MT][LWMAX] in RP-tile chunks
// (the batch rules of rs_gather). args at a0: nu, (b | slot << 8 | chip << 16, mesh) x nu, nc, (x | y << 8 | k << 16,
// rows: 4 bits a block - r0 | nr << 2) x nc, LAT address, word address, LATZ address, counter address, LWMAX. Chunk
// m: its LAT at + m LA_CHL bytes, the words' targets count up (seen, the caller's), re-armed after the last
FORCE_INLINE void la_gather(uint32_t a0, uint32_t& pos, uint32_t m, uint32_t* seen) {
    const uint32_t nu = get_arg_val<uint32_t>(a0), u0 = a0 + 1, nc = get_arg_val<uint32_t>(u0 + 2 * nu);
    if (nu == 0) return;
    const uint32_t c0 = u0 + 2 * nu + 1, t0a = c0 + 2 * nc;
    const uint32_t lat = get_arg_val<uint32_t>(t0a) + m * LA_CHL, word = get_arg_val<uint32_t>(t0a + 1), LW = get_arg_val<uint32_t>(t0a + 4);
    if (m == 0)
        for (uint32_t i = 0; i < nc; ++i) seen[i] = 0;
    uint32_t c = 0, need = 0, wptr = 0;  // (pos: chunks pushed since the phases began - the CB's batch alignment)
    auto flush = [&]() {
        noc_async_read_barrier();
        cb_push_back(0, c * RP);
        pos += c;
        c = 0;
    };
    for (uint32_t un = 0; un < nu; ++un) {
        const uint32_t b = get_arg_val<uint32_t>(u0 + 2 * un) & 0xFF;
        for (uint32_t i = 0; i < nc; ++i) {
            const uint32_t cw = get_arg_val<uint32_t>(c0 + 2 * i), rw = (get_arg_val<uint32_t>(c0 + 2 * i + 1) >> (4 * b)) & 0xF;
            const uint32_t r0 = rw & 3, nr = rw >> 2, cnt = nr * LW;
            if (cnt == 0) continue;
            volatile tt_l1_ptr uint32_t* mb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word + XG_HS * (cw >> 16));
            const uint32_t target = ++seen[i];
            invalidate_l1_cache();
            HWW(21, 0xA5000000u | un << 8 | i); HWW(22, target); HWW(23, *mb);
            if (*mb < target && c > 0) flush();
            noc_semaphore_wait_min(mb, target);
            const uint64_t src = get_noc_addr(cw & 0xFF, (cw >> 8) & 0xFF, lat + r0 * LW * 2048);
            for (uint32_t t0 = 0; t0 < cnt; t0 += RP) {
                if (c == 0) {
                    need = GROUP - pos % GROUP;
                    cb_reserve_back(0, need * RP);
                    wptr = get_write_ptr(0);
                }
                const uint32_t n = cnt - t0 < RP ? cnt - t0 : RP;
                noc_async_read(src + t0 * 2048, wptr + c * RP * 2048, n * 2048);
                if (++c == need) flush();
            }
        }
        if (c > 0) flush();
    }
    if (m + 1 < NCH) return;
    for (uint32_t i = 0; i < nc; ++i)
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word + XG_HS * (get_arg_val<uint32_t>(c0 + 2 * i) >> 16)) = 0;
}
"""

_LA_SENDER = r"""
// (K3's latent all-to-all) the reader's walk: chunks into chip b's same core (multi-hop fabric unicast: a header
// routed to b a unit), LATZ slot of my chip, its rows; a core's last chunk with the fused increment of its counter
template <typename C, typename H, typename L>
FORCE_INLINE void la_send(uint32_t a0, C& conn, H* hdrs, L&& lpoll, uint32_t& rpos, uint32_t m) {
    const uint32_t nu = get_arg_val<uint32_t>(a0), u0 = a0 + 1, nc = get_arg_val<uint32_t>(u0 + 2 * nu);
    if (nu == 0) return;
    const uint32_t c0 = u0 + 2 * nu + 1, t0a = c0 + 2 * nc;
    const uint32_t latz = get_arg_val<uint32_t>(t0a + 2) + m * LA_CHL, lctr = get_arg_val<uint32_t>(t0a + 3), LW = get_arg_val<uint32_t>(t0a + 4);
    {  // (end-to-end fence: every destination chip started this call - its LATZ is free; my neighbour's ready only
       // fences the first hop)
        const uint32_t fence = get_arg_val<uint32_t>(t0a + 5), need = get_arg_val<uint32_t>(t0a + 6);
        HWW(24, 0xD0000000u); HWW(25, need); HWW(26, *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fence));
        while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fence) < need) {
            invalidate_l1_cache();
            lpoll();
        }
    }
    uint32_t u = 0, lim = GROUP - rpos % GROUP;
    auto pop_batch = [&]() {
        noc_async_writes_flushed();
        cb_pop_front(0, u * RP);
        rpos += u;
        u = 0;
        lim = GROUP - rpos % GROUP;
    };
    for (uint32_t un = 0; un < nu; ++un) {
        const uint32_t w0 = get_arg_val<uint32_t>(u0 + 2 * un), b = w0 & 0xFF, slot = (w0 >> 8) & 0xFF;
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[GROUP + 1 + un];
        for (uint32_t i = 0; i < nc; ++i) {
            const uint32_t cw = get_arg_val<uint32_t>(c0 + 2 * i), rw = (get_arg_val<uint32_t>(c0 + 2 * i + 1) >> (4 * b)) & 0xF;
            const uint32_t r0 = rw & 3, cnt = (rw >> 2) * LW;
            if (cnt == 0) continue;
            const uint32_t x = cw & 0xFF, y = (cw >> 8) & 0xFF;
            const uint32_t dst0 = latz + (slot * MT + r0) * LW * 2048;
            for (uint32_t t0 = 0; t0 < cnt; t0 += RP) {
                lpoll();
                while (!cb_pages_available_at_front(0, RP * (u + 1))) lpoll();
                cb_wait_front(0, RP * (u + 1));
                const uint32_t src = get_read_ptr(0) + u * RP * 2048;
                const uint32_t n = cnt - t0 < RP ? cnt - t0 : RP;
                const uint64_t dst = get_noc_addr(x, y, dst0 + t0 * 2048);
                noc_async_writes_flushed();  // (this unit's one header: its previous packet has left)
                if (t0 + RP >= cnt) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(NocUnicastAtomicIncFusedCommandHeader{dst, get_noc_addr(x, y, lctr), 1, true}, n * 2048);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, n * 2048);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(src, n * 2048, reinterpret_cast<uint32_t>(hdr));
                if (++u == lim) pop_batch();
            }
        }
        if (u > 0) pop_batch();
    }
}
"""

_RS_SENDER = r"""
// (down + reduce-scatter) my units' chunks, the reader's walk: one fabric packet each into the downstream chip's same
// core, its zone slot of this direction (slot by the core's half); a core's last chunk carries the fused increment of
// its arrival counter. Pops at the reader's batch boundaries. args at r0: as the reader's, then the zone base, the
// arrival counter's address, P (zone slot tiles; 0: compact zones - a core's slot at the sum of its tiles of the
// blocks in the unit's prefix mask for the core's half, w0 bits 20 + 4 half)
template <typename C, typename H, typename L>
FORCE_INLINE void rs_send(uint32_t r0, C& conn, H* hdrs, L&& lpoll, uint32_t& rpos, uint32_t m) {
    const uint32_t nu = get_arg_val<uint32_t>(r0), u0 = r0 + 1, nc = get_arg_val<uint32_t>(u0 + 2 * nu);
    if (nu == 0) return;
    const uint32_t c0 = u0 + 2 * nu + 1, en0 = c0 + 2 * nc;
    const uint32_t zb = get_arg_val<uint32_t>(en0 + G + 2), actr = get_arg_val<uint32_t>(en0 + G + 3);
    const uint32_t P = get_arg_val<uint32_t>(en0 + G + 4);
    if (m > 0) {  // (chunk m into the downstream's zones once its cores read chunk m - 1's: its port's credits)
        volatile tt_l1_ptr uint32_t* zc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(en0 + G + 6));
        while (*zc < m) {
            invalidate_l1_cache();
            lpoll();
        }
    }
    uint32_t u = 0, lim = GROUP - rpos % GROUP, h = 0;
#ifdef SE_PROBE_RSSTAT
    DeviceZoneScopedN("SE-RSS");
    auto clk = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    uint32_t win = 0, wslot = 0, npk = 0;  // (cycles waiting for the reader's data; for a free fabric slot)
#endif
    auto pop_batch = [&]() {
        noc_async_writes_flushed();
        cb_pop_front(0, u * RP);
        rpos += u;
        u = 0;
        lim = GROUP - rpos % GROUP;
    };
    for (uint32_t un = 0; un < nu; ++un) {
        const uint32_t w0 = get_arg_val<uint32_t>(u0 + 2 * un), w1 = get_arg_val<uint32_t>(u0 + 2 * un + 1);
        const uint32_t b = (w0 >> 8) & 0xFF, mask = (w0 >> 16) & 3;
        for (uint32_t i = 0; i < nc; ++i) {
            const uint32_t cw = get_arg_val<uint32_t>(c0 + 2 * i), cnts = get_arg_val<uint32_t>(c0 + 2 * i + 1);
            const uint32_t cnt = (cnts >> (8 * b)) & 0xFF, hf = (cw >> 16) & 1;
            if (!((mask >> hf) & 1) || cnt == 0) continue;
            const uint32_t x = cw & 0xFF, y = (cw >> 8) & 0xFF;
            uint32_t zo;
            if (P) {
                zo = (hf ? (w1 >> 16) : (w1 & 0xFFFF)) * P;
            } else {
                const uint32_t pm = (w0 >> (20 + 4 * hf)) & 0xF;
                zo = 0;
                for (uint32_t j = 0; j < 4; ++j) {
                    if ((pm >> j) & 1) zo += (cnts >> (8 * j)) & 0xFF;
                }
            }
            const uint32_t dst0 = zb + zo * 2048;
            const uint64_t ctr = get_noc_addr(x, y, actr);
            for (uint32_t t0 = 0; t0 < cnt; t0 += RP) {
                HWW(24, 0xC1000000u | un << 8 | i); HWW(25, t0); HWW(26, u); HWW(27, rpos);
                lpoll();  // (the logits / routing go between my packets: ahead of my next one)
#ifdef SE_PROBE_RSSTAT
                {
                    const uint32_t t = clk();
                    while (!cb_pages_available_at_front(0, RP * (u + 1))) lpoll();
                    win += clk() - t;
                }
#endif
                while (!cb_pages_available_at_front(0, RP * (u + 1))) lpoll();
                cb_wait_front(0, RP * (u + 1));
                const uint32_t src = get_read_ptr(0) + u * RP * 2048;
                const uint32_t n = cnt - t0 < RP ? cnt - t0 : RP;
                const uint64_t dst = get_noc_addr(x, y, dst0 + t0 * 2048);
                volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                h = h + 1 == GROUP ? 0 : h + 1;
                if (t0 + RP >= cnt) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(NocUnicastAtomicIncFusedCommandHeader{dst, ctr, 1, true}, n * 2048);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, n * 2048);
                }
#ifdef SE_PROBE_RSSTAT
                {
                    const uint32_t t = clk();
                    conn.wait_for_empty_write_slot();
                    wslot += clk() - t;
                    if (npk++ == 0) DeviceTimestampedData("SE-RSS-T1", 1);  // (the first RS packet's time)
                }
#endif
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(src, n * 2048, reinterpret_cast<uint32_t>(hdr));
                if (++u == lim) pop_batch();
            }
        }
        if (u > 0) pop_batch();
    }
#ifdef SE_PROBE_RSSTAT
    DeviceTimestampedData("SE-RSS-WIN", win);
    DeviceTimestampedData("SE-RSS-WSLOT", wslot);
    DeviceTimestampedData("SE-RSS-NPK", npk);
#endif
}
"""

_READER = (
    '#include <cstdint>\n#include "api/dataflow/dataflow_api.h"\n'
    + _CT
    + "constexpr auto x_args = TensorAccessorArgs<15>();\n"
    + _WALK
    + _LA_READER
    + _RS_READER
    + r"""
void kernel_main() {
    size_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++), xown = get_arg_val<uint32_t>(a++), v = get_arg_val<uint32_t>(a++);
    const uint32_t arrival = get_arg_val<uint32_t>(a++), n = get_arg_val<uint32_t>(a++), up_ipc = get_arg_val<uint32_t>(a++);
    const uint32_t ent = a, T = ent + 4 * n;  // entries (q, part, up_base within a chunk, packets a chunk); tables
    // (after the tables: the engine's "own shard loaded" counter here and its full count)
    const uint32_t xl0 = T + G + S0 + S1 + 3 * GX + GY;
    volatile tt_l1_ptr uint32_t* xloaded = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(xl0));
    const uint32_t xl_need = get_arg_val<uint32_t>(xl0 + 1);
    bool xl = false;
    const auto x = TensorAccessor(x_args, x_addr, 2048);
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival);
    uint32_t batch = 0, wptr = 0;
    for (uint32_t m = 0; m < NCH; ++m)
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t q = get_arg_val<uint32_t>(ent + 4 * k), part = get_arg_val<uint32_t>(ent + 4 * k + 1);
            const uint32_t up_base = get_arg_val<uint32_t>(ent + 4 * k + 2), npk = get_arg_val<uint32_t>(ent + 4 * k + 3);
            uint32_t done = 0;
            walk(m, q, part, v, [&](uint32_t R, uint32_t j, uint32_t kj, uint32_t kk0, uint32_t np, uint32_t idx) {
                if (batch == 0) {
                    cb_reserve_back(0, RP * GROUP);
                    wptr = get_write_ptr(0);
                }
                const uint32_t dst = wptr + batch * RP * 2048;
                if (k == 0) {  // my own shard: from the engine's x-own L1 once every engine core loaded it (no DRAM here:
                               // the DRAM is the engine's weight load's)
                    if (!xl) {
                        noc_semaphore_wait_min(xloaded, xl_need);
                        xl = true;
                    }
                    const Dst d = dst_of(T, xown, m, R, j, kk0);
                    noc_async_read(get_noc_addr(d.x, d.y, d.addr), dst, np * XTB);
                } else {  // relay: upstream's packet idx of (chunk m, its entry k - 1) landed here
                    noc_semaphore_wait_min(arrived, m * up_ipc + up_base + idx / INC + 1);
                    const Dst d = dst_of(T, xown, m, R, j, kk0);
                    noc_async_read(get_noc_addr(d.x, d.y, d.addr), dst, np * XTB);
                }
                ++batch;
                ++done;
                if (batch == GROUP || done == npk) {
                    noc_async_read_barrier();
                    cb_push_back(0, RP * GROUP);
                    batch = 0;
                }
            });
        }
    noc_semaphore_wait_min(xloaded, xl_need);  // (re-arm: every engine core counted itself on every port)
    noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(xloaded)), 0u - xl_need);
    noc_async_atomic_barrier();
    const uint32_t rsa = xl0 + 2, rnu = get_arg_val<uint32_t>(rsa), rnc = get_arg_val<uint32_t>(rsa + 1 + 2 * rnu);
    uint32_t pos = 0;  // (the gather left the CB batch-aligned; the phases after it share one chunk count)
    uint32_t seen[128], lseen[128];
    for (uint32_t m = 0; m < NCH; ++m) {
        HWW(20, 0xB1);
        la_gather(rsa + 2 + 2 * rnu + 2 * rnc + G + RSX, pos, m, lseen);
        HWW(20, 0xB2);
        rs_gather(rsa, pos, m, seen);
        HWW(20, 0xB3);
    }
}
"""
)

_SENDER = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "api/debug/kernel_profiler.hpp"
using namespace tt::tt_fabric;
"""
    + _CT
    + _WALK
    + _LA_SENDER
    + _RS_SENDER
    + LSTREAM
    + RA_SEND
    + r"""
void kernel_main() {
    size_t a = 0;
    const uint32_t xown = get_arg_val<uint32_t>(a++), v = get_arg_val<uint32_t>(a++);
    const uint32_t arrival = get_arg_val<uint32_t>(a++), expect_in = get_arg_val<uint32_t>(a++);  // (my counter)
    const uint32_t ready = get_arg_val<uint32_t>(a++), wait_ready = get_arg_val<uint32_t>(a++);
    const uint32_t send_ready = get_arg_val<uint32_t>(a++), rx = get_arg_val<uint32_t>(a++), ry = get_arg_val<uint32_t>(a++);
    const uint32_t px = get_arg_val<uint32_t>(a++), py = get_arg_val<uint32_t>(a++);  // downstream port (NoC)
    const uint16_t mesh = get_arg_val<uint32_t>(a++), chip = get_arg_val<uint32_t>(a++);  // the neighbour
    const uint32_t wx = get_arg_val<uint32_t>(a++), wy = get_arg_val<uint32_t>(a++), watch = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t ent = a, T = ent + 3 * n;  // entries (q, part, packets a chunk); tables
    a = T + G + S0 + S1 + 3 * GX + GY;  // (past the tables: the RS units, then the fabric connection)
    const uint32_t rs0 = a, rnu = get_arg_val<uint32_t>(rs0), rnc = get_arg_val<uint32_t>(rs0 + 1 + 2 * rnu);
    const uint32_t ls0 = rs0 + 2 + 2 * rnu + 2 * rnc + G + RSX;
    // (chunks: the zone credits - my zone-free word (my chip's cores, a chunk's arrivals from my peer read), the
    // signals a chunk; forwarded to the peer's port facing me (its zone-credit word): NCH - 1 a call)
    const uint32_t rse = rs0 + 2 + 2 * rnu + 2 * rnc + G;
    volatile tt_l1_ptr uint32_t* zfree = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(rse + 7));
    const uint32_t zneed = get_arg_val<uint32_t>(rse + 8), zexp = get_arg_val<uint32_t>(rse + 9);
    const uint32_t zcred = get_arg_val<uint32_t>(rse + 6);
    const uint32_t NZF = zneed && NCH > 1 ? NCH - 1 : 0;
    uint32_t zfw = 0;
    LStream ls;
    ls.init(ls0);
    const uint32_t la0 = ls0 + LStream::len(ls0), lnu = get_arg_val<uint32_t>(la0);
    // (the latent's end-to-end fence: my fence word, the "started" count my sends need (one a destination chip), then
    // the chips I tell "started" - their latent ports toward me: (chip | mesh << 16, x | y << 8))
    const uint32_t lf0 = la0 + 2 + 2 * lnu + 2 * get_arg_val<uint32_t>(la0 + 1 + 2 * lnu) + 5;
    const uint32_t lfence = get_arg_val<uint32_t>(lf0), lneed = get_arg_val<uint32_t>(lf0 + 1), lns = get_arg_val<uint32_t>(lf0 + 2);
    const uint32_t ra0 = lf0 + 3 + 2 * lns;
    RASend ra;
    ra.init(ra0);
    const uint32_t rf0 = ra0 + RASend::len(ra0);
    RAFwd rf;
    rf.init(rf0);
    a = rf0 + RAFwd::len(rf0);
    if (n > 0 || send_ready || rnu > 0 || ls.nu > 0 || lnu > 0 || lns > 0 || !ra.done || rf.left || NZF) {
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[GROUP + 1 + 3];
        for (uint32_t h = 0; h <= GROUP; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
            (void)fabric_set_unicast_route(hdrs[h], chip, mesh);
        }
        volatile tt_l1_ptr PACKET_HEADER_TYPE* rahdr = PacketHeaderPool::allocate_header();  // (routing: routed a packet)
        volatile tt_l1_ptr PACKET_HEADER_TYPE* lshdr = PacketHeaderPool::allocate_header();  // (logits: routed a unit)
        for (uint32_t un = 0; un < lnu && un < 3; ++un) {  // (the latent's destinations: routed a unit)
            const uint32_t w0 = get_arg_val<uint32_t>(la0 + 1 + 2 * un);
            hdrs[GROUP + 1 + un] = PacketHeaderPool::allocate_header();
            (void)fabric_set_unicast_route(hdrs[GROUP + 1 + un], static_cast<uint16_t>(w0 >> 16),
                                           static_cast<uint16_t>(get_arg_val<uint32_t>(la0 + 2 + 2 * un)));
        }
        conn.open();
        if (send_ready) {  // fence: I started this call - the neighbour's port facing me may write into my L1
            hdrs[GROUP]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(rx, ry, ready), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdrs[GROUP]), sizeof(PACKET_HEADER_TYPE));
        }
        if (lns > 0) {  // (I started this call: every chip sending me latent may - its latent port's fence word)
            volatile tt_l1_ptr PACKET_HEADER_TYPE* fh = PacketHeaderPool::allocate_header();
            for (uint32_t i = 0; i < lns; ++i) {
                const uint32_t w = get_arg_val<uint32_t>(lf0 + 3 + 2 * i), xy = get_arg_val<uint32_t>(lf0 + 4 + 2 * i);
                (void)fabric_set_unicast_route(fh, static_cast<uint16_t>(w & 0xFFFF), static_cast<uint16_t>(w >> 16));
                fh->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(xy & 0xFF, (xy >> 8) & 0xFF, lfence), 1, true});
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(fh), sizeof(PACKET_HEADER_TYPE));
            }
        }
        if (wait_ready) {
            DeviceZoneScopedN("SE-RDYWAIT");  // (its end: the neighbour's port started - launch skew + a hop)
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready), 1);
            noc_semaphore_inc(get_noc_addr(ready), 0u - 1u);
            noc_async_atomic_barrier();
        }
        const uint64_t arr_noc = get_noc_addr(px, py, arrival);
        uint32_t h = 0, unflushed = 0;
#ifndef RA_N_
#define RA_N_ 1
#define RA_M_ 1
#define RA_EPC_ 1
#define RA_ME_ 0
#define RA_HBP_ 64
#define RA_BB_ 16
#define RA_BBP_ 16
#define RA_MAXP_ 64
#endif
        auto zpoll = [&]() {  // (my chip's cores read a chunk's zones: credit the peer's sends, a blocking header -
            // the gather's fence / watcher header, never in flight across this)
            if (zfw < NZF && *zfree >= (zfw + 1) * zneed) {
                hdrs[GROUP]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(rx, ry, zcred), 1, true});
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdrs[GROUP]), sizeof(PACKET_HEADER_TYPE));
                ++zfw;
            }
        };
        auto lpoll = [&]() {
            zpoll();
            ls.poll(conn, lshdr, false);
            ra.poll(conn, rahdr, false, RA_N_, RA_M_, RA_EPC_, RA_ME_, RA_HBP_, RA_BB_, RA_BBP_, RA_MAXP_);
            rf.poll(conn, rahdr, false, RA_N_, RA_M_, RA_HBP_, RA_MAXP_);
        };
        for (uint32_t m = 0; m < NCH && n > 0; ++m) {
            for (uint32_t k = 0; k < n; ++k) {
                const uint32_t q = get_arg_val<uint32_t>(ent + 3 * k), part = get_arg_val<uint32_t>(ent + 3 * k + 1);
                const uint32_t npk = get_arg_val<uint32_t>(ent + 3 * k + 2);
                uint32_t sent = 0;
                walk(m, q, part, v, [&](uint32_t R, uint32_t j, uint32_t, uint32_t kk0, uint32_t np, uint32_t) {
#if LS_FIRST
                    lpoll();  // (GLM-type: the logits ahead of the x gather)
                    while (!cb_pages_available_at_front(0, RP * (unflushed + 1))) lpoll();
#endif
                    cb_wait_front(0, RP * (unflushed + 1));
                    const uint32_t src = get_read_ptr(0) + unflushed * RP * 2048;
                    const Dst d = dst_of(T, xown, m, R, j, kk0);
                    const uint64_t dst = get_noc_addr(d.x, d.y, d.addr);
                    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[h];
                    h = h + 1 == GROUP ? 0 : h + 1;
                    if ((++sent % INC) == 0 || sent == npk) {
                        hdr->to_noc_fused_unicast_write_atomic_inc(NocUnicastAtomicIncFusedCommandHeader{dst, arr_noc, 1, true}, np * XTB);
                    } else {
                        hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, np * XTB);
                    }
                    conn.wait_for_empty_write_slot();
                    conn.send_current_slot_non_blocking(src, np * XTB, reinterpret_cast<uint32_t>(hdr));
                    if (++unflushed == GROUP || sent == npk) {
                        noc_async_writes_flushed();
                        cb_pop_front(0, RP * GROUP);
                        unflushed = 0;
                    }
                });
                // this chunk's shard k all sent: the neighbour's watcher (behind its packets on this link)
                hdrs[GROUP]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(wx, wy, watch), 1, true});
                conn.wait_for_empty_write_slot();
                conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdrs[GROUP]), sizeof(PACKET_HEADER_TYPE));
            }
        }
        lpoll();
        uint32_t rpos = 0;  // (as the reader's pos)
        for (uint32_t m = 0; m < NCH; ++m) {
            HWW(28, 0xE1);
            la_send(la0, conn, hdrs, lpoll, rpos, m);
            HWW(28, 0xE2);
            rs_send(rs0, conn, hdrs, lpoll, rpos, m);
            HWW(28, 0xE3);
        }
        while (zfw < NZF) {  // (the chunks' credits still owed)
            invalidate_l1_cache();
            lpoll();
        }
        ls.poll(conn, lshdr, true);  // (whatever is left of the logits)
        ra.poll(conn, rahdr, true, RA_N_, RA_M_, RA_EPC_, RA_ME_, RA_HBP_, RA_BB_, RA_BBP_, RA_MAXP_);  // (the routing)
        rf.poll(conn, rahdr, true, RA_N_, RA_M_, RA_HBP_, RA_MAXP_);  // (and what passes through)
        noc_async_writes_flushed();
        if (NZF) noc_semaphore_inc(get_noc_addr(reinterpret_cast<uint32_t>(zfree)), 0u - NZF * zneed);  // (re-arm)
        if (zexp) {  // (every credit of the call came: they were waited for, the last before the last chunk)
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zcred), zexp);
            noc_semaphore_inc(get_noc_addr(zcred), 0u - zexp);
        }
        ls.rearm();
        if (lneed) noc_semaphore_inc(get_noc_addr(lfence), 0u - lneed);  // (re-arm: a next call's early "started" stays)
        conn.close();
    }
    if (expect_in > 0) {  // everything from upstream landed; re-arm for the next call
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival);
        noc_semaphore_wait_min(arrived, expect_in);
        noc_semaphore_inc(get_noc_addr(arrival), 0u - expect_in);
        noc_async_atomic_barrier();
    }
}
"""
)

_WATCHER = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#ifdef W_RA0
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
#include "api/debug/kernel_profiler.hpp"
"""
    + RA_SEND
    + r"""
#endif
// chunk m's shard of rank k (k = 1 .. G - 1 in the engine's order) is in once each stream carrying it counted it
// (a stream counts every shard it sends: m entries + its index + 1): then XREADY + 1 on every engine core
// (XREADY = m (G - 1) + k). args: chunks, the XREADY id, ranks G - 1, engine cores nc, counters ncnt; (counter address,
// final) x ncnt; cores (x, y) x nc; per rank: nreq, (counter address, the entry's index, the stream's entries a chunk)
void kernel_main() {
    size_t a = 0;
    const uint32_t nch = get_arg_val<uint32_t>(a++), sem = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t nk = get_arg_val<uint32_t>(a++), nc = get_arg_val<uint32_t>(a++), ncnt = get_arg_val<uint32_t>(a++);
    const uint32_t cnt0 = a, c0 = cnt0 + 2 * ncnt, r0 = c0 + 2 * nc;
    for (uint32_t m = 0; m < nch; ++m) {
        uint32_t b = r0;
        for (uint32_t k = 0; k < nk; ++k) {
            const uint32_t nreq = get_arg_val<uint32_t>(b++);
            for (uint32_t i = 0; i < nreq; ++i, b += 3)
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(b)),
                                       m * get_arg_val<uint32_t>(b + 2) + get_arg_val<uint32_t>(b + 1) + 1);
            for (uint32_t c = 0; c < nc; ++c)
                noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(c0 + 2 * c), get_arg_val<uint32_t>(c0 + 2 * c + 1), sem), 1);
        }
    }
    for (uint32_t i = 0; i < ncnt; ++i)  // (re-arm)
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(cnt0 + 2 * i)), 0u - get_arg_val<uint32_t>(cnt0 + 2 * i + 1));
    noc_async_atomic_barrier();
#ifdef W_RA0
    {  // (the routing exchange's second Y direction: after the watch)
        RASend rs;
        rs.init(W_RA0);
        size_t a2 = W_RA0 + RASend::len(W_RA0);
        if (get_arg_val<uint32_t>(a2++)) {
            a2 += 3;  // (the Y hello's args: the BRISC's direction only)
            auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a2);
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
            conn.open();
            rs.poll(conn, hdr, true, RA_N_, RA_M_, RA_EPC_, RA_ME_, RA_HBP_, RA_BB_, RA_BBP_, RA_MAXP_);
            conn.close();
        }
        noc_async_atomic_barrier();
    }
#endif
}
"""
)


def _incs(n):
    return n // INC_EVERY + (1 if n % INC_EVERY else 0)


def chunk_packets(CR, nv, v, part, sps, kb, rp):
    """Packets of one chunk's shard entry on link v (the kernels' walk: chunk-local row lr over link (lr / 2) % nv,
    part 1 / 2 its even / odd rows; a row tile's K steps in pieces of rp tiles)."""
    rows = sum(1 for lr in range(CR) if (lr // 2) % nv == v and (part == 0 or (lr & 1) == part - 1))
    return rows * sps * -(-kb // rp)


def port_plan(chips, coord, d, v, *, CR, nv, sps, kb, rp, nch):
    """One port's stream (pure: the device build and the CPU protocol model share it). Returns dict(peer, up,
    mine=[(shard, part, upstream base within a chunk, packets a chunk)], up_ipc (the upstream's increments a chunk),
    expect_in (a call), has_up)."""
    rg = chips[coord]["rings"][0]
    peer = rg["next"] if d == "fwd" else rg["prev"]
    up = rg["prev"] if d == "fwd" else rg["next"]
    mine = rg[d] if peer is not None else []
    ups = chips[up]["rings"][0][d] if up is not None else []
    up_base, acc = [], 0
    for _, part in ups:
        up_base.append(acc)
        acc += _incs(chunk_packets(CR, nv, v, part, sps, kb, rp))
    ent = [
        (q, part, up_base[k - 1] if k > 0 else 0, chunk_packets(CR, nv, v, part, sps, kb, rp))
        for k, (q, part) in enumerate(mine)
    ]
    return dict(peer=peer, up=up, mine=ent, up_ipc=acc, expect_in=nch * acc if ups else 0, has_up=bool(ups))


def shard_order(chips, coord, *, CR, nv, sps, kb, rp):
    """(pure: the device build and the CPU model share it) chip coord's shards in the engine's processing order - its
    own, then by arrival (the latest entry index carrying it, then the shard) - and per remote shard the (direction,
    link, entry index, entries a chunk) of the incoming streams carrying a part of it."""
    rg = chips[coord]["rings"][0]
    own = (rg["fwd"] or rg["bwd"])[0][0]
    carry = {}
    for d in ("fwd", "bwd"):  # data flowing d arrives from the d-upstream chip
        up = rg["prev"] if d == "fwd" else rg["next"]
        if up is None:
            continue
        ups = chips[up]["rings"][0][d]
        for j, (q, part) in enumerate(ups):
            for v in range(nv):
                if chunk_packets(CR, nv, v, part, sps, kb, rp):
                    carry.setdefault(q, []).append((d, v, j, len(ups)))
    rest = sorted(carry, key=lambda q: (max(j for _, _, j, _ in carry[q]), q))
    return [own] + rest, carry


def port_cb_bytes(kb, group=4):
    """A port core's static circular buffer (the sender's packets: two groups of rp pages) at K step kb."""
    rp = max(1, min(kb, ttnn.get_tt_fabric_max_payload_size_bytes() // 2048))
    return 2 * group * rp * 2048


class XGather3:
    """The row-chunk-major x all-gather into an SeEngine3's x-own (module docstring); movers on sub_core_grid."""

    def __init__(
        self,
        mesh_device,
        engine,
        *,
        num_links=2,
        group=4,
        ring=None,
        sub_core_grid=None,
        placement="auto",
        watcher=None,
    ):
        """watcher: the watcher's logical core (default: the plan's first copy core)."""
        self.mesh, self.eng = mesh_device, engine
        e = engine
        G = self.G = tuple(mesh_device.shape)[1]
        assert e.KT % G == 0 and (e.KT // G) % e.KB == 0, "a shard must be whole K steps"
        self.klt = e.KT // G
        self.sps = self.klt // e.KB
        self.rp = max(1, min(e.KB, ttnn.get_tt_fabric_max_payload_size_bytes() // 2048))
        assert port_cb_bytes(e.KB, group) == 2 * group * self.rp * 2048  # (the engine's arena cap counts on it)
        assert group + 1 <= 12, "the sender takes group + 1 packet headers (12 a RISC)"
        self.group, self.nv = group, num_links
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
        for ch in self.chips.values():
            for d in ("fwd", "bwd"):
                assert all(part == 0 for _, part in ch["rings"][0][d][:-1]), "a relay's upstream entry must be whole"
        self._watcher = watcher
        cores = sorted(
            {(c.x, c.y) for ch in self.chips.values() for c in ch["ports"].values()}
            | {(self.watcher_core(k).x, self.watcher_core(k).y) for k in self.chips}
        )
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])
        self.sems = [persist_l1.semaphore(mesh_device, crs, 0) for _ in range(3 + 2 * num_links)]
        addrs = [persist_l1.address(s) for s in self.sems]
        self.arrival, self.ready, self.xloaded = addrs[0], addrs[1], addrs[-1]
        self.watch = {
            (d, v): addrs[2 + i * num_links + v] for i, d in enumerate(("fwd", "bwd")) for v in range(num_links)
        }

    def ring_dir(self, p, b):
        """The ring direction ("fwd" / "bwd") a multi-hop packet from TP position p to b leaves by: the shorter way, and
        on a tie (the ring's far side) the way the fabric's own routing sends it - a packet injected toward the other
        neighbour is not delivered (a hang: the latent all-to-all on a ring of 4). Positions, the same on every row
        (asserted)."""
        G = self.G
        df = (b - p) % G
        if df != G - df:
            return "fwd" if df < G - df else "bwd"
        if not hasattr(self, "_ties"):
            node = lambda c: self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(*c))  # noqa: E731
            self._ties = {}
            for coord, ch in self.chips.items():
                rg = ch["rings"][0]
                pp = rg["p"]
                at = {self.chips[c]["rings"][0]["p"]: c for c in self.chips if c[0] == coord[0]}
                fd = ttnn._ttnn.fabric.get_eth_forwarding_direction(node(coord), node(rg["next"]))
                for q, qc in at.items():
                    if (q - pp) % G * 2 == G:
                        d = (
                            "fwd"
                            if ttnn._ttnn.fabric.get_eth_forwarding_direction(node(coord), node(qc)) == fd
                            else "bwd"
                        )
                        assert self._ties.setdefault((pp, q), d) == d, "ring ties differ between mesh rows"
        return self._ties[(p, b)]

    def _pk(self, v, part):
        """Packets of a chunk's shard entry on link v."""
        e = self.eng
        rows = sum(1 for R in range(e.CR) if (R // 2) % self.nv == v and (part == 0 or (R & 1) == part - 1))
        return rows * self.sps * -(-e.KB // self.rp)

    def port_cores(self, coord):
        """This chip's port cores (NoC): the engine counts its own-shard x load on each (xloaded)."""
        virt = self.mesh.worker_core_from_logical_core
        return [(virt(c).x, virt(c).y) for c in self.chips[coord]["ports"].values()]

    def watcher_core(self, coord):
        return self._watcher if self._watcher is not None else self.chips[coord]["copy"][0]

    def order(self, coord):
        """(this chip's shards in the engine's order, per remote shard its carrying streams): shard_order."""
        return shard_order(self.chips, coord, CR=self.eng.CR, nv=self.nv, sps=self.sps, kb=self.eng.KB, rp=self.rp)

    def ranks(self, coord):
        shards, _ = self.order(coord)
        r = [0] * self.G
        for k, q in enumerate(shards):
            r[q] = k
        return r

    def tables(self, coord):
        """The owner tables (dst_of's) for chip coord's L1: its shard ranks, family 0 / 1 columns, per column (nown,
        family-1 base slot), NoC x / y."""
        e, mesh = self.eng, self.mesh
        nf = e.NSG // 2
        per = []
        for c in range(e.GX):
            n0 = nf // e.S0 if c in e.fam0 else 0
            n1 = nf // e.S1 if c in e.fam1 else 0
            per += [n0 + n1, n0]
        vx = [mesh.worker_core_from_logical_core(ttnn.CoreCoord(e.cx0 + c, 0)).x for c in range(e.GX)]
        vy = [mesh.worker_core_from_logical_core(ttnn.CoreCoord(e.cx0, cy)).y for cy in range(e.GY)]
        return self.ranks(coord) + list(e.fam0) + list(e.fam1) + per + vx + vy

    def rs_block(self, rs, d, v):
        """A port's down + reduce-scatter args (se_rs): units, its link's cores, the entry order, FINB / mailbox,
        the downstream zone of direction d, its arrival counter, P. rs: dict(plan, p, finb, mbox, zone, arr)."""
        plan, p, e = rs["plan"], rs["p"], self.eng
        virt = lambda cx, cy: self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(e.cx0 + cx, cy))  # noqa: E731
        units = plan.port_units(p, d) if plan.down(p, d) is not None else []
        cores = plan.port_cores(v) if units else []
        q = plan.down(p, d)
        # (compact zones: each half's prefix mask - the blocks of the downstream's slots before this unit's)
        pm = lambda b, h, s_: plan.prefix_mask(q, h, d, s_) if plan.compact else 0  # noqa: E731
        out = [len(units)] + [
            w
            for eu, b, m, s0, s1 in units
            for w in (eu | b << 8 | m << 16 | pm(b, 0, s0) << 20 | pm(b, 1, s1) << 24, s0 | s1 << 16)
        ]
        out += [len(cores)]
        for cx, cy in cores:
            c, k = virt(cx, cy), plan.k(cx, cy)
            out += [c.x | c.y << 8 | plan.half(k) << 16 | k << 20, plan.counts_word(cx, cy)]
        i = DIRS_I[d]
        # (chunks: zone credits - my chip's cores taking arrivals from the peer this port faces (opposite direction)
        # signal here a chunk; the peer's port facing me (direction opposite) sends me its cores' signals)
        opp = "bwd" if d == "fwd" else "fwd"
        # (the cores something lands on - a core with only "nothing comes" slots does not signal)
        lands = lambda q_, cx, cy, d_: any(t for _, _, t in plan.arrivals(q_, cx, cy)[d_])  # noqa: E731
        zneed = sum(1 for cx, cy in plan.port_cores(v) if lands(p, cx, cy, opp)) if e.NCH > 1 else 0
        zexp = 0
        q = plan.down(p, d)
        if e.NCH > 1 and q is not None:
            zexp = e.NCH - 1 if any(lands(q, cx, cy, d) for cx, cy in plan.port_cores(v)) else 0
        return (
            out
            + plan.entries(p)
            + [rs["finb"], rs["mbox"], rs["zone"][i], rs["arr"][i], 0 if plan.compact else plan.P]
            + [rs["ffree"], rs["zcred"], rs["zfree"], zneed, zexp]
        )

    def la_block(self, rs, la, d, v):
        """A port's latent all-to-all args (la_gather / la_send): its destination blocks, its link's cores with their
        rows a block, the engine's LAT / words / LATZ / counter, LWMAX. la: dict(word, ctr) (None: no latent)."""
        e = self.eng
        import os

        if not la or not getattr(e, "LT", None) or "NOA2A" in os.environ.get("SE_PROBE", ""):
            return [0, 0] + [0] * 5 + [0, 0, 0]
        plan, p = rs["plan"], rs["p"]
        node = lambda pos: self._node_of_pos(rs["coord"], pos)  # noqa: E731
        dst = [b for b in range(self.G) if b != p and e.la_dir(p, b) == d]
        units = []
        for b in dst:
            n = node(b)
            units += [b | (p if p < b else p - 1) << 8 | int(n.chip_id) << 16, int(n.mesh_id)]
        assert len(dst) <= 3 and self.group + 1 + len(dst) <= 12, "packet headers: 12 a RISC"
        virt = lambda cx, cy: self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(e.cx0 + cx, cy))  # noqa: E731
        cores = [(cx, cy) for cy in range(e.GY) for cx in range(e.GX) if (cy * e.GX + cx) % self.nv == v]
        cw = []
        for cx, cy in cores:
            c = virt(cx, cy)
            rows = 0
            for b in range(self.G):
                r0, nr = e.la_rows(cy, b)
                rows |= (r0 | nr << 2) << (4 * b)
            cw += [c.x | c.y << 8 | (cy * e.GX + cx) << 16, rows]
        # (the end-to-end fence: my sends wait one "started" a destination chip; I tell every chip whose latent comes
        # to me through this direction's link v - its port (its direction to me, v))
        sends = []
        for q in range(self.G):
            if q == p or e.la_dir(p, q) != d:
                continue
            qc = next(c for c, ch in self.chips.items() if c[0] == rs["coord"][0] and ch["rings"][0]["p"] == q)
            pc = {(dd, l): c for (_, dd, l), c in self.chips[qc]["ports"].items()}[(e.la_dir(q, p), v)]
            pv = self.mesh.worker_core_from_logical_core(pc)
            nq = node(q)
            sends += [int(nq.chip_id) | int(nq.mesh_id) << 16, pv.x | pv.y << 8]
        return (
            [len(dst)]
            + units
            + [len(cores)]
            + cw
            + [e.base + e.off["lat"], la["word"], e.base + e.off["latz"], la["ctr"], e.LWMAX]
            + [la["fence"], len(dst), len(sends) // 2]
            + sends
        )

    def _node_of_pos(self, coord, pos):
        """The fabric node of the chip at TP position pos in coord's row."""
        for c, ch in self.chips.items():
            if c[0] == coord[0] and ch["rings"][0]["p"] == pos:
                return self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(*c))
        raise KeyError(pos)

    def pieces(self, coord, prog, x, xready_sem, rs=None, router=None, la=None, routing=None):
        """(CBs, kernels) of chip `coord` (fabric connections set up on prog); x: this chip's [rows, kl] in DRAM.
        rs: the down + reduce-scatter phase's (rs_block)."""
        mesh, chips, e = self.mesh, self.chips, self.eng
        virt = lambda c: mesh.worker_core_from_logical_core(c)  # noqa: E731
        node = lambda c: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*c))  # noqa: E731
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
            processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
        )  # noqa: E731
        st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
        cs = lambda cores: ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])  # noqa: E731
        ch = chips[coord]
        rg = ch["rings"][0]
        ports = {(d, l): core for (_, d, l), core in ch["ports"].items()}
        xown = e.base + e.off["xown"]
        T = self.tables(coord)
        cbs = [
            ttnn.CBDescriptor(
                total_size=2 * self.group * self.rp * 2048,
                core_ranges=cs(list(ports.values())),
                format_descriptors=[ttnn.CBFormatDescriptor(CB, ttnn.bfloat16, 2048)],
            )
        ]
        rd_rt, sd_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        streams_in = []
        for (d, v), core in ports.items():
            pl = port_plan(chips, coord, d, v, CR=e.CR, nv=self.nv, sps=self.sps, kb=e.KB, rp=self.rp, nch=e.NCH)
            peer, mine, acc = pl["peer"], pl["mine"], pl["up_ipc"]
            if pl["has_up"]:
                streams_in.append(self.watch[(d, v)])
            ent_r = [x_ for q, part, ub, npk in mine for x_ in (q, part, ub, npk)]
            ent_s = [x_ for q, part, ub, npk in mine for x_ in (q, part, npk)]
            rd_rt[core.x][core.y] = (
                [int(x.buffer_address()), xown, v, self.arrival, len(mine), acc]
                + ent_r
                + T
                + [self.xloaded, e.GX * e.GY]
                + (self.rs_block(rs, d, v) if rs else [0, 0] + [0] * (self.G + RSX))
                + self.la_block(rs, la, d, v)
            )
            rsb = self.rs_block(rs, d, v) if rs else [0, 0] + [0] * (self.G + RSX)
            rsb = rsb + (router.port_block(coord, d, v) if router else [0, 0] + [0] * 9)
            lab = self.la_block(rs, la, d, v)
            assert lab[0] == 0 or mine, "the latent needs the gather's fence"
            rab = routing.send_block(coord, ("x", d, v)) if routing is not None else [0]
            rab = rab + (routing.fwd_block(coord, ("x", d, v)) if routing is not None else [0])
            assert rab[0] == 0 or mine or not rab[1], "the routing's fabric packets need the gather's fence"
            rsb = rsb + lab + rab
            assert rsb[0] == 0 or mine, "a port sending reduce-scatter units must have the gather's fence"
            assert not (router and router.port_block(coord, d, v)[0]) or mine, "the logits need the gather's fence"
            args = [xown, v, self.arrival, pl["expect_in"], self.ready, int(bool(mine))]
            if peer is not None and rg.get(f"{d}_links"):
                pp = {(dd, ll): cc for (_, dd, ll), cc in chips[peer]["ports"].items()}
                rc, pc, wc = virt(pp[(_OPP[d], v)]), virt(pp[(d, v)]), virt(self.watcher_core(peer))
                pn = node(peer)
                args += (
                    [
                        int(bool(chips[peer]["rings"][0][_OPP[d]])),
                        rc.x,
                        rc.y,
                        pc.x,
                        pc.y,
                        int(pn.mesh_id),
                        int(pn.chip_id),
                        wc.x,
                        wc.y,
                        self.watch[(d, v)],
                        len(mine),
                    ]
                    + ent_s
                    + self.tables(peer)
                    + rsb
                )
                args += list(ttnn.setup_fabric_connection(node(coord), pn, rg[f"{d}_links"][v], prog, core))
            else:
                assert rsb[0] == 0 and not (router and router.port_block(coord, d, v)[0])
                args += [0] * 10 + [0] + T + rsb
            sd_rt[core.x][core.y] = args
        ct = [
            e.KB,
            e.MT,
            self.sps,
            self.rp,
            e.CR,
            e.NCH,
            self.nv,
            self.klt,
            self.group,
            INC_EVERY,
            e.GX,
            e.GY,
            e.S0,
            e.S1,
            self.G,
        ]
        x_ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=f"#define XG_HS {persist_l1.handoff_bytes()}u\n"
                + f"#define LA_CHL {getattr(e, 'LATP', 0) * 2048 if e.NCH > 1 else 0}u\n"
                + f"#define XTB {getattr(e, 'xtb', 2048)}u\n"  # (x's tile bytes: the engine's x_dtype; the CB keeps 2048 B slots)
                + _probes()
                + _READER,
                source_type=st,
                core_ranges=cs(list(ports.values())),
                compile_time_args=ct + x_ct,
                runtime_args=rd_rt,
                config=dm("RISCV_0", NOC1) if PORT_NOC0 else dm("RISCV_1", NOC0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"#define XG_HS {persist_l1.handoff_bytes()}u\n#define LS_FIRST {int(bool(router and router.prio_first))}\n"
                + f"#define LA_CHL {getattr(e, 'LATP', 0) * 2048 if e.NCH > 1 else 0}u\n"
                + f"#define XTB {getattr(e, 'xtb', 2048)}u\n"
                + (routing.defines(coord) if routing is not None else "")
                + _probes()
                + _SENDER,
                source_type=st,
                core_ranges=cs(list(ports.values())),
                compile_time_args=ct,
                runtime_args=sd_rt,
                config=dm("RISCV_1", NOC0) if PORT_NOC0 else dm("RISCV_0", NOC1),
            ),
        ]
        ecores = [virt(ttnn.CoreCoord(e.cx0 + c, cy)) for c in range(e.GX) for cy in range(e.GY)]
        wt = ttnn.RuntimeArgs()
        wc = self.watcher_core(coord)
        shards, carry = self.order(coord)
        finals, wargs = {}, []
        for q in shards[1:]:
            wargs += [len(carry[q])] + [x_ for d, v, j, n in carry[q] for x_ in (self.watch[(d, v)], j, n)]
        for d in ("fwd", "bwd"):  # (a stream counts all its entries, also those carrying nothing on this link)
            up = rg["prev"] if d == "fwd" else rg["next"]
            if up is not None and chips[up]["rings"][0][d]:
                for v in range(self.nv):
                    finals[self.watch[(d, v)]] = e.NCH * len(chips[up]["rings"][0][d])
        wl = (
            [e.NCH, xready_sem, len(shards) - 1, len(ecores), len(finals)]
            + [x_ for k_, f in finals.items() for x_ in (k_, f)]
            + [v_ for c in ecores for v_ in (c.x, c.y)]
            + wargs
        )
        wsrc = _WATCHER
        if routing is not None:
            wdef, wra = routing.watcher_tail(coord, prog)
            if wra:
                wsrc = wdef + f"#define W_RA0 {len(wl)}\n" + _WATCHER
                wl = wl + wra
        wt[wc.x][wc.y] = wl
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=f"#define XG_HS {persist_l1.handoff_bytes()}u\n" + wsrc,
                source_type=st,
                core_ranges=cs([wc]),
                compile_time_args=[],
                runtime_args=wt,
                config=dm("RISCV_1", NOC1),
            )
        )
        return cbs, kernels
