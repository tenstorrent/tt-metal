// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert store-and-forward forwarder, NCRISC (NOC1; NOC0 with RCfg.swap_noc): owns the fabric connection to the
// next chip of this directed ring and walks the ring's order table (wr_seq.hpp, loaded into CB 2 by the BRISC) in
// order:
//   INJ:  sends the chunk the BRISC read into CB 0;
//   FWD:  once staging slot k % S has arrived, sends it on;
//   TERM: nothing to send (the BRISC lands it).
// A send is one fabric hop: a fused write + atomic inc into the downstream forwarder's staging slot (sent % S) and its
// arrival word, issued once downstream has freed that slot (sent - W_DFREE < S). Every packet terminates one hop away.
// Staging credits: slot k is free once this RISC has sent it (or it was TERM) and the BRISC has landed it; the freed
// count goes to the opposite-direction forwarder on this chip (its W_RELAY word), whose fabric connection leads to
// this ring's upstream chip, and it sends the count on as a fabric inline write into upstream's W_DFREE word. This
// core does the same for its pair: W_RELAY changes go to the next chip, batched by CRB while packets are flowing.
// Every wait polls the relay, so credits keep moving while this ring is blocked.
// Deadlock freedom: the tables are the time order of a synchronous model of all rings (ws_sf.conveyor) in which each
// packet goes on the moment it arrives, and a packet of a group's expert n reaches each chip only after every packet
// of the group's experts up to n - 1 (the group cores free expert n - 2's slot once expert n - 1's weights are in).
// Each wait here is on an event earlier in that model's time: the previous table entry, the packet's previous hop,
// the downstream slot freed S packets earlier (S > the model's hop time), and for landing the group slot of expert
// n - 2, freed once the chip has every slice of the experts up to n - 1: ring packets that arrived earlier, and the
// chip's own slices, which its local lander takes in the order of the model time after which their slot can be freed
// (ws_sf.local_items). So no cycle of waits can form, whatever the real timing.
// At the end the core keeps relaying until its pair's last count went out and downstream returned all its slots, so
// no fabric write lands after the program ends.
// RT: 0 table length, 1 packets in (FWD + TERM), 2 packets out (INJ + FWD), 3 packets in on the pair's ring,
//     4 my mesh id, 5 next chip id, 6 next chip's forwarder (same ring) xy, 7 pair xy (this chip), 8 next chip's pair
//     xy, then fabric args
// CT: 0 NEXP, 1 KT, 2 KTD, 3 DCOLS, 4 CHT, 5 G, 6 GC, 7 NCHIP, 8 S, 9 words, 10 CRB
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#include "wr_seq.hpp"
using namespace tt::tt_fabric::mesh::experimental;

void kernel_main() {
    constexpr uint32_t CHT = get_compile_time_arg_val(4), S = get_compile_time_arg_val(8);
    constexpr uint32_t words = get_compile_time_arg_val(9), CRB = get_compile_time_arg_val(10);

    const uint32_t len = get_arg_val<uint32_t>(0), total_in = get_arg_val<uint32_t>(1);
    const uint32_t total_out = get_arg_val<uint32_t>(2), pair_in = get_arg_val<uint32_t>(3);
    const uint16_t my_mesh = get_arg_val<uint32_t>(4);
    const uint8_t next_chip = get_arg_val<uint32_t>(5);
    const uint32_t dxy = get_arg_val<uint32_t>(6), pxy = get_arg_val<uint32_t>(7), rxy = get_arg_val<uint32_t>(8);
    size_t arg_idx = 9;
    auto conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
    conn.open();
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hcr = PacketHeaderPool::allocate_header();

    const uint32_t tb = get_tile_size(0);
    const uint32_t CHB = CHT * tb;
    // (staging is CB 1: the same address on every forwarder core of every chip, since all have the same CBs)
    const uint32_t stage = get_write_ptr(1);
    const uint64_t down_arr = get_noc_addr(dxy & 0xFF, dxy >> 8, words + wr::W_ARR * 16);
    const uint64_t pair_relay = get_noc_addr(pxy & 0xFF, pxy >> 8, words + wr::W_RELAY * 16);
    const uint64_t up_pair_dfree = get_noc_addr(rxy & 0xFF, rxy >> 8, words + wr::W_DFREE * 16);
    auto rd = [](uint32_t w) {
        invalidate_l1_cache();
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + w * 16);
    };

    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    uint32_t t_arr = 0, t_dcred = 0, t_cb = 0, t_send = 0, t_drain = 0, n_relay = 0, t_first = 0, t_last = 0;
    const uint32_t tstart = wall();

    uint32_t k_in = 0, ndone = 0, sent = 0, free_sent = 0, relayed = 0, ti = 0;
#ifdef WS_DIAG
    // progress counters for a hang post-mortem, readable as CB 3 / 4 pushed / popped (triage dumps CB counters):
    // CB 3 pushed = table entries done, popped = packets sent; CB 4 pushed = W_DFREE seen, popped = W_ARR seen
    uint32_t dg[4] = {0, 0, 0, 0};
    auto diag = [&](uint32_t dfree, uint32_t arrv) {
        const uint32_t v[4] = {ti, sent, dfree, arrv};
        for (uint32_t q = 0; q < 4; ++q) {
            if (v[q] > dg[q]) {
                if (q % 2 == 0) {
                    cb_push_back(3 + q / 2, v[q] - dg[q]);
                } else {
                    cb_pop_front(3 + q / 2, v[q] - dg[q]);
                }
                dg[q] = v[q];
            }
        }
    };
#endif
    // With WS_DEFER a send is issued without waiting for its payload to leave L1: that wait (settle) comes before the
    // next send, or while this RISC waits on anything, and only then does the packet's staging slot count as done
    // (FWD) or its CB 0 chunk get popped (INJ); without it every send settles at once. With WS_DEFER the L1 inline
    // writes also go through rotating scratch words (a plain inline write flushes every outstanding write, the
    // payload included), and the credit header waits for a flush only when its previous packet may still be in L1.
    bool pend = false, pend_inj = false, cr_out = false;
    uint32_t pend_k = 0, n_inj = 0, scr_n = 0, scr_since = 0;
    const uint32_t inj_base = get_write_ptr(0);
    auto flushed = [&]() {
        noc_async_writes_flushed();
        scr_since = 0;
        cr_out = false;
    };
    auto settle = [&]() {
        if (pend) {
            flushed();
            if (pend_inj) {
                cb_pop_front(0, CHT);
            }
            pend = false;
        }
    };
    auto inline_l1 = [&](uint64_t dst, uint32_t v) {
#ifdef WS_DEFER
        if (scr_since == 8) {
            flushed();
        }
        const uint32_t scr = words + (8 + scr_n % 8) * 16;
        ++scr_n;
        ++scr_since;
        noc_inline_dw_write<InlineWriteDst::L1, false, false>(dst, v, 0xF, noc_index, NOC_UNICAST_WRITE_VC, scr);
#else
        noc_inline_dw_write(dst, v);
#endif
    };
    auto update_free = [&]() {
        const uint32_t landed = rd(wr::W_LANDED);
        const uint32_t nd = pend && !pend_inj ? pend_k : ndone;
        const uint32_t f = landed < nd ? landed : nd;
        if (f > free_sent) {
            inline_l1(pair_relay, f);
            free_sent = f;
        }
    };
    auto service = [&](bool idle) {
        if (idle) {
            settle();
        }
        update_free();
#ifdef WS_DIAG
        diag(rd(wr::W_DFREE), rd(wr::W_ARR));
#endif
        const uint32_t r = rd(wr::W_RELAY);
        if (r > relayed && (idle || r - relayed >= CRB || r == pair_in)) {
#ifdef WS_DEFER
            if (cr_out) {
                flushed();  // (the credit header's previous packet has left L1)
            }
#else
            flushed();
#endif
            fabric_unicast_noc_unicast_inline_write(
                &conn,
                hcr,
                next_chip,
                my_mesh,
                tt::tt_fabric::NocUnicastInlineWriteCommandHeader{up_pair_dfree, r});
            cr_out = true;
            relayed = r;
            ++n_relay;
        }
    };
    auto send = [&](uint32_t src, bool inj) {
        const uint32_t c0 = wall();
        while (sent - rd(wr::W_DFREE) >= S) {
            service(true);
        }
        const uint32_t c1 = wall();
        t_dcred += c1 - c0;
        if (sent == 0) {
            t_first = c1;
        }
        settle();  // (the previous payload and its header are out of L1)
        fabric_unicast_noc_fused_unicast_with_atomic_inc(
            &conn,
            hdr,
            next_chip,
            my_mesh,
            src,
            CHB,
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                get_noc_addr(dxy & 0xFF, dxy >> 8, stage + (sent % S) * CHB), down_arr, 1, true});
        pend = true;
        pend_inj = inj;
#ifndef WS_DEFER
        settle();
#endif
        ++sent;
        t_last = wall();
        t_send += t_last - c1;
    };

    while (rd(wr::W_TABLE) == 0) {
    }
    volatile tt_l1_ptr uint32_t* T = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(2));
    for (uint32_t i = 0; i < len; ++i, ti = i) {
        const uint32_t kind = T[i] & 3;
        if (kind == wr::INJ) {
            // (chunk n_inj sits at slot n_inj % SCB of CB 0, as the BRISC reads them; a pending one is not popped yet)
            const uint32_t a0 = wall();
            while (!cb_pages_available_at_front(0, CHT * (pend && pend_inj ? 2 : 1))) {
                service(true);
            }
            t_cb += wall() - a0;
            send(inj_base + (n_inj % SCB_CHUNKS) * CHB, true);
            ++n_inj;
        } else if (kind == wr::FWD) {
            const uint32_t a0 = wall();
            while (rd(wr::W_ARR) < k_in + 1) {
                service(true);
            }
            t_arr += wall() - a0;
            send(stage + (k_in % S) * CHB, false);
            pend_k = k_in;
            ndone = ++k_in;
        } else {
            ndone = ++k_in;
        }
        service(false);
    }
    settle();
    const uint32_t d0 = wall();
    while (free_sent < total_in || relayed < pair_in || rd(wr::W_DFREE) < sent) {
        service(true);
    }
    t_drain = wall() - d0;
    noc_async_writes_flushed();
    conn.close();
    noc_async_full_barrier();
    volatile tt_l1_ptr uint32_t* sst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SST_ADDR);
    sst[0] = t_arr;
    sst[1] = t_dcred;
    sst[2] = t_cb;
    sst[3] = t_send;
    sst[4] = t_drain;
    sst[5] = wall() - tstart;
    sst[6] = sent;
    sst[7] = total_in;
    sst[8] = pair_in;
    sst[9] = n_relay;
    sst[10] = conn.num_buffers_per_channel;
    sst[11] = stage;
    sst[12] = t_last - t_first;
    sst[13] = total_out;
}
