// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert all-chip weight landing, owner-side sender (NCRISC). Every chip owns experts e % NCHIP == own and lands
// their slices on every chip's group cores (same logical cores everywhere): one core a fabric branch (E-only,
// W-only, N/S with E/W teeth; one 2D multicast reaches every chip of the branch) and, with WF_LOCAL, one core that
// lands them on this chip by NoC writes. For each owned expert e (group g = e % G, ring slot n & 1, n = e / G) and each
// core c of group g: waits until every covered chip k freed the slot (credit word k NC + c >= n - 1, written by chip
// k's relay), then sends the slice from CB 0 (filled by wf_sreader) in CHT-tile packets into the core's ring; the last
// packet is fused with an atomic inc of the core's warr word for the slot. Two cores a branch (links 0 and 1, targets
// c % 2 == link); the link-0 core also forwards the relay's credit entries (WF_CREDQ) between its packets.
// RT: 0 my mesh id, 1 my chip id, 2 e hops, 3 w hops, 4 n hops, 5 s hops, 6 covered chips (bit mask), 7 ntargets,
//     8.. targets (c | x << 8 | y << 16), then (not WF_LOCAL) the fabric connection args
// CT: 0 NEXP, 1 KT, 2 KTD, 3 DCOLS, 4 CHT, 5 G, 6 GC, 7 wgu_ring, 8 wd_ring, 9 warr, 10 icred, 11 NC, 12 NCHIP, 13 own
#include "api/dataflow/dataflow_api.h"
#ifndef WF_LOCAL
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
using namespace tt::tt_fabric::mesh::experimental;
#endif

void kernel_main() {
    constexpr uint32_t NEXP = get_compile_time_arg_val(0), KT = get_compile_time_arg_val(1);
    constexpr uint32_t KTD = get_compile_time_arg_val(2), DCOLS = get_compile_time_arg_val(3);
    constexpr uint32_t CHT = get_compile_time_arg_val(4), G = get_compile_time_arg_val(5);
    constexpr uint32_t GC = get_compile_time_arg_val(6);
    constexpr uint32_t wgu_ring = get_compile_time_arg_val(7), wd_ring = get_compile_time_arg_val(8);
    constexpr uint32_t warr = get_compile_time_arg_val(9), icred = get_compile_time_arg_val(10);
    constexpr uint32_t NC = get_compile_time_arg_val(11), NCHIP = get_compile_time_arg_val(12);
    constexpr uint32_t own = get_compile_time_arg_val(13);
    constexpr uint32_t WGU = KT * 2, WDN = KTD * DCOLS, WTOT = WGU + WDN;
    const uint32_t tb = get_tile_size(0);
    const uint32_t CHB = CHT * tb;

    const uint32_t covered = get_arg_val<uint32_t>(6);
    const uint32_t nt = get_arg_val<uint32_t>(7);
#ifndef WF_LOCAL
    const uint16_t my_mesh = get_arg_val<uint32_t>(0);
    const uint8_t my_chip = get_arg_val<uint32_t>(1);
    const MeshMcastRange ranges{
        (uint8_t)get_arg_val<uint32_t>(2),
        (uint8_t)get_arg_val<uint32_t>(3),
        (uint8_t)get_arg_val<uint32_t>(4),
        (uint8_t)get_arg_val<uint32_t>(5)};
    size_t arg_idx = 8 + nt;
    auto conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
    conn.open();
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr[2] = {
        PacketHeaderPool::allocate_header(), PacketHeaderPool::allocate_header()};
    uint32_t hi = 0;
#endif
#ifdef WF_CREDQ
    // credit queue (this direction's link-0 sender): entries {seq, o | c << 8, dst core xy, v} from the relay, sent on
    // as fabric inline writes of v into chip o's sender word my_chip * NC + c; a {.., 0xFFFFFFFF} entry ends it
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hcr = PacketHeaderPool::allocate_header();
    volatile tt_l1_ptr uint32_t* q = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(WF_CREDQ);
    uint32_t rp = 0;
    bool qdone = false;
    auto drain = [&]() {
        invalidate_l1_cache();
        while (q[(rp % WF_NQ) * 4] == rp + 1) {
            const uint32_t w1 = q[(rp % WF_NQ) * 4 + 1], xy = q[(rp % WF_NQ) * 4 + 2], v = q[(rp % WF_NQ) * 4 + 3];
            ++rp;
            if (w1 == 0xFFFFFFFF) {
                qdone = true;
                continue;
            }
            fabric_unicast_noc_unicast_inline_write(
                &conn,
                hcr,
                (uint8_t)(w1 & 0xFF),
                my_mesh,
                tt::tt_fabric::NocUnicastInlineWriteCommandHeader{
                    get_noc_addr(xy & 0xFF, xy >> 8, icred + (my_chip * NC + (w1 >> 8)) * 16), v});
            noc_async_writes_flushed();
            invalidate_l1_cache();
        }
    };
#else
    auto drain = []() {};
#endif
    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    uint32_t t_cb = 0, t_send = 0, t_cred = 0;
    const uint32_t tstart = wall();

    for (uint32_t e = own; e < NEXP; e += NCHIP) {
        const uint32_t g = e % G, n = e / G;
        for (uint32_t t = 0; t < nt; ++t) {
            const uint32_t a = get_arg_val<uint32_t>(8 + t);
            const uint32_t c = a & 0xFF, tx = (a >> 8) & 0xFF, ty = (a >> 16) & 0xFF;
            if (c / GC != g) {
                continue;
            }
            if (n >= 2) {  // every covered chip freed expert n - 2's slot of core c
                const uint32_t a0 = wall();
                for (uint32_t k = 0; k < NCHIP; ++k) {
                    if (!((covered >> k) & 1)) {
                        continue;
                    }
                    volatile tt_l1_ptr uint32_t* cr =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(icred + (k * NC + c) * 16);
                    do {
                        drain();
                        invalidate_l1_cache();
                    } while (*cr < n - 1);
                }
                t_cred += wall() - a0;
            }
            for (uint32_t u0 = 0; u0 < WTOT; u0 += CHT) {
                const uint32_t dst =
                    u0 < WGU ? wgu_ring + ((n & 1) * WGU + u0) * tb : wd_ring + ((n & 1) * WDN + u0 - WGU) * tb;
                const uint32_t a1 = wall();
                while (!cb_pages_available_at_front(0, CHT)) {
                    drain();
                }
                cb_wait_front(0, CHT);
                const uint32_t a2 = wall();
                t_cb += a2 - a1;
                const bool last = u0 + CHT >= WTOT;
#ifdef WF_LOCAL
                noc_async_write(get_read_ptr(0), get_noc_addr(tx, ty, dst), CHB);
                if (last) {
                    noc_async_write_barrier();
                    noc_semaphore_inc(get_noc_addr(tx, ty, warr + (n & 1) * 16), 1);
                } else {
                    noc_async_writes_flushed();
                }
#else
                volatile tt_l1_ptr PACKET_HEADER_TYPE* h = hdr[hi];
                hi ^= 1;
                if (!last) {
                    fabric_multicast_noc_unicast_write(
                        &conn,
                        h,
                        my_chip,
                        my_mesh,
                        ranges,
                        get_read_ptr(0),
                        CHB,
                        tt::tt_fabric::NocUnicastCommandHeader{get_noc_addr(tx, ty, dst)});
                } else {
                    fabric_multicast_noc_fused_unicast_with_atomic_inc(
                        &conn,
                        h,
                        my_chip,
                        my_mesh,
                        ranges,
                        get_read_ptr(0),
                        CHB,
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            get_noc_addr(tx, ty, dst), get_noc_addr(tx, ty, warr + (n & 1) * 16), 1, true});
                }
                noc_async_writes_flushed();  // (payload read out of CB 0 and the header out of L1)
#endif
                t_send += wall() - a2;
                cb_pop_front(0, CHT);
                drain();
            }
        }
    }
#ifdef WF_CREDQ
    while (!qdone) {  // (other chips' credits still pass through this connection)
        drain();
    }
#endif
#ifndef WF_LOCAL
    conn.close();
#endif
    noc_async_full_barrier();
    volatile tt_l1_ptr uint32_t* sst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SST_ADDR);
    sst[0] = t_cb;
    sst[1] = t_send;
    sst[2] = t_cred;
    sst[3] = wall() - tstart;
}
