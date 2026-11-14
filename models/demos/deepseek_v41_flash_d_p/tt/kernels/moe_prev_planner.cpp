// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// MOE_REPLICAS=prev (P.56): the replica planner of one MoE layer, run on every chip right after the layer's MoE call
// (tt/expert_prev.py has the terms, the host mirror and the data layouts). ROLE 1 (sender cores, one a first fabric hop
// direction): this chip's expert counts (its own slots of head.counts) to every other chip's landing row, a fused
// write + atomic increment of the receiver's arrival counter. ROLE 0 (the planner core): waits for the other chips'
// counts, plans the layer's replicas for its next call from them (the same integer arithmetic on every chip: every
// chip gets the same plan), writes the slot map into the layer's router bias, the copies of its own chip into the fill
// job pages of the coming MoE calls (the combine's REC_FX pages, the copied layer's buffers in words 6 / 7), and the
// state. ROLE 2 (reset, a request's first chunk): every job page header of this chip zeroed, the global state
// emptied, a new epoch (every layer's state then reads as empty).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/fabric_common.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;

constexpr uint32_t ROLE = get_compile_time_arg_val(0);
// P.62 (MOE_PREV_HOST2=lean): 1 - ONE program for the whole mesh (DS41-225's pattern): a chip finds its position from
// the fabric routing table's my_device_id and a table in its runtime args (planner: after the ring args; sender: the
// mesh layout below), so every chip runs the same runtime args; 0 - one program a chip, the position an arg (P.56)
constexpr uint32_t MESH1 = get_compile_time_arg_val(21);
// the planner's constants and the planner itself (ROLE 0): the fork's header, shared with the fused head's watcher
// tail (P.62b, MOE_PLANNER=folded); compile args 1 .. 20 (expert_prev.Geometry.ct)
#define PP_CT0 0
#include "ttnn/ttnn/bringup/moe_block_main_ttnn/fused_head/device/kernels/prev_planner.hpp"
using namespace pp;

// ring slot s's job tensor (its address: runtime arg a0 + s, or a0 alone)
inline uint32_t ring_at(uint32_t a0, uint32_t s) { return get_arg_val<uint32_t>(RINGPG ? a0 : a0 + s); }

// byte k of the runtime args from a0 (4 a word, little end first)
inline uint32_t arg_byte(uint32_t a0, uint32_t k) {
    return (get_arg_val<uint32_t>(a0 + k / 4) >> (8 * (k % 4))) & 0xFFu;
}

// 2-bit field k of the runtime args from a0 (16 a word, low bits first)
inline uint32_t arg_2bit(uint32_t a0, uint32_t k) {
    return (get_arg_val<uint32_t>(a0 + k / 16) >> (2 * (k % 16))) & 3u;
}

// MESH1: this chip's fabric chip id (the routing table in L1, as mhc_post_feed_common.hpp's Table)
inline uint32_t my_dev() {
    return reinterpret_cast<tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE)->my_device_id;
}

inline void route_to(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0))
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
#error "the replica planner needs a 2D fabric (agent rule 4)"
#endif
}

void kernel_main() {
    const uint32_t base = al64(get_write_ptr(0));
    if constexpr (ROLE == 1 && MESH1) {
        // ---------------------------------------------------------------- sender (one mesh program, P.62)
        // args: 0 counts, 1 landing, 2 global state, 3 arrival sem, 4 / 5 the planner core's x / y, 6 this core's
        // direction d, 7 mesh id, 8 table bytes of me (fabric chip id -> position), then 9 .. 11 the fabric connection
        // (9: the eth channel, overwritten from the table), then at 12 the tables (bytes, 4 a word): me[NDEV words x 4]
        // at TME, chip id of each position [NC] at TCH, first hop direction of (me, k) [NC x NC] at THP (two
        // bits each, 16 a word; (me, me) unused), the eth channel of (me, d) [NC x 4] at TCN (0xFF: no port)
        const uint32_t cnt = get_arg_val<uint32_t>(0), land = get_arg_val<uint32_t>(1), gst = get_arg_val<uint32_t>(2);
        const uint32_t sem = get_arg_val<uint32_t>(3), px = get_arg_val<uint32_t>(4), py = get_arg_val<uint32_t>(5);
        const uint32_t d = get_arg_val<uint32_t>(6), mesh_id = get_arg_val<uint32_t>(7);
        const uint32_t ndev = get_arg_val<uint32_t>(8);
        constexpr uint32_t TME = 12;
        const uint32_t TCH = TME + (ndev + 3) / 4, THP = TCH + (NC + 3) / 4, TCN = THP + (NC * NC + 15) / 16;
        const uint32_t me = arg_byte(TME, my_dev());
        uint32_t nd = 0;
        for (uint32_t k = 0; k < NC; ++k) {
            nd += (k != me && arg_2bit(THP, me * NC + k) == d) ? 1u : 0u;
        }
        if (nd == 0) {
            return;  // (no target through this direction from my chip: no connection opened)
        }
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(9)) = arg_byte(TCN, me * 4 + d);
        size_t fa = 9;
        const uint32_t l_cnt = base, l_g = al64(l_cnt + CPAGE), l_pay = l_g + 64;
        read_page(gst, GW * 4, 0, l_g, 64);
        read_page(cnt, CPAGE, 0, l_cnt, CPAGE);
        noc_async_read_barrier();
        invalidate_l1_cache();
        const volatile tt_l1_ptr uint32_t* cw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_cnt);
        volatile tt_l1_ptr uint32_t* pay = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_pay);
        for (uint32_t j = 0; j < SPW; ++j) {
            pay[j] = j < SPC ? cw[me * SPC + j] : 0u;
        }
        const uint32_t par = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_g)[0] & 1;
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(fa);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
        conn.open();
        const InterleavedAddrGen<true> lg = {.bank_base_address = land, .page_size = SPW * 4};
        const uint64_t to = lg.get_noc_addr(par * NC + me);  // (my row on every receiver: the same address)
        const uint64_t arrival = get_noc_addr(px, py, sem);
        for (uint32_t k = 0; k < NC; ++k) {
            if (k == me || arg_2bit(THP, me * NC + k) != d) {
                continue;
            }
            route_to(hdr, static_cast<uint16_t>(arg_byte(TCH, k)), static_cast<uint16_t>(mesh_id));
            hdr->to_noc_fused_unicast_write_atomic_inc(
                NocUnicastAtomicIncFusedCommandHeader{to, arrival, 1, true}, SPW * 4);
            conn.wait_for_empty_write_slot();
            conn.send_current_slot_non_blocking(l_pay, SPW * 4, reinterpret_cast<uint32_t>(hdr));
            noc_async_writes_flushed();  // (the payload and the header left L1 before the header is rewritten)
        }
        conn.close();
        return;
    }
    if constexpr (ROLE == 1 && !MESH1) {
        // ---------------------------------------------------------------- sender: my counts to the other chips
        const uint32_t cnt = get_arg_val<uint32_t>(0), land = get_arg_val<uint32_t>(1), gst = get_arg_val<uint32_t>(2);
        const uint32_t me = get_arg_val<uint32_t>(3), sem = get_arg_val<uint32_t>(4);
        const uint32_t px = get_arg_val<uint32_t>(5), py = get_arg_val<uint32_t>(6), nd = get_arg_val<uint32_t>(7);
        size_t fa = 8 + 2 * nd;
        const uint32_t l_cnt = base, l_g = al64(l_cnt + CPAGE), l_pay = l_g + 64;
        read_page(gst, GW * 4, 0, l_g, 64);
        read_page(cnt, CPAGE, 0, l_cnt, CPAGE);
        noc_async_read_barrier();
        invalidate_l1_cache();
        const volatile tt_l1_ptr uint32_t* cw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_cnt);
        volatile tt_l1_ptr uint32_t* pay = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_pay);
        for (uint32_t j = 0; j < SPW; ++j) {
            pay[j] = j < SPC ? cw[me * SPC + j] : 0u;
        }
        const uint32_t par = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_g)[0] & 1;
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(fa);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
        conn.open();
        const InterleavedAddrGen<true> lg = {.bank_base_address = land, .page_size = SPW * 4};
        const uint64_t to = lg.get_noc_addr(par * NC + me);  // (my row on every receiver: the same address)
        const uint64_t arrival = get_noc_addr(px, py, sem);
        for (uint32_t j = 0; j < nd; ++j) {
            const uint16_t mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(8 + 2 * j));
            const uint16_t chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(9 + 2 * j));
            route_to(hdr, chip_id, mesh_id);
            hdr->to_noc_fused_unicast_write_atomic_inc(
                NocUnicastAtomicIncFusedCommandHeader{to, arrival, 1, true}, SPW * 4);
            conn.wait_for_empty_write_slot();
            conn.send_current_slot_non_blocking(l_pay, SPW * 4, reinterpret_cast<uint32_t>(hdr));
            noc_async_writes_flushed();  // (the payload and the header left L1 before the header is rewritten)
        }
        conn.close();
        return;
    }
    if constexpr (ROLE == 2) {
        // ---------------------------------------------------------------- reset: a request starts
        const uint32_t gst = get_arg_val<uint32_t>(0);
        const uint32_t l_g = base, l_z = al64(l_g + GW * 4);
        volatile tt_l1_ptr uint32_t* G = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_g);
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_z);
        read_page(gst, GW * 4, 0, l_g, GW * 4);
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t i = 0; i < 16; ++i) {
            z[i] = 0;
        }
        for (uint32_t s = 0; s < W; ++s) {
            const uint32_t ring = ring_at(2, s);
            for (uint32_t d = 0; d < 4; ++d) {
                write_page(ring, PG, ring_page(s, d * LNK), 0, l_z, 64);
            }
        }
        G[1] = G[1] + 1;  // (a new epoch: every layer's state reads as empty)
        for (uint32_t s = 0; s < W; ++s) {
            G[GSTAMP + s] = NO_EXPERT;
        }
        for (uint32_t i = GBUD; i < GW; ++i) {
            G[i] = 0;
        }
        write_page(gst, GW * 4, 0, 0, l_g, GW * 4);
        noc_async_write_barrier();
        return;
    }
    // -------------------------------------------------------------------- planner (prev_planner.hpp)
    constexpr uint32_t RING0 = 10;
    // (args: counts (0: folded - the own row from the landing rows, a plan only when one is pending), landing, global
    // state, layer state, router bias, gate/up, down, directions; 8 the position (MESH1: unused - the table of
    // positions by fabric chip id follows the ring args), 9 the arrival counter, 10 .. the ring)
    const uint32_t me = MESH1 ? arg_byte(RING0 + (RINGPG ? 1 : W), my_dev()) : get_arg_val<uint32_t>(8);
    const Io io{
        .cnt = get_arg_val<uint32_t>(0),
        .land = get_arg_val<uint32_t>(1),
        .gst = get_arg_val<uint32_t>(2),
        .sta = get_arg_val<uint32_t>(3),
        .rb = get_arg_val<uint32_t>(4),
        .gu = get_arg_val<uint32_t>(5),
        .dw = get_arg_val<uint32_t>(6),
        .cst = get_arg_val<uint32_t>(7),
        .me = me,
        .sem = get_arg_val<uint32_t>(9),
        .ring = get_arg_val<uint32_t>(RING0),
        .ring_arg0 = RING0,
        .base = base};
    (void)plan(io);
}
