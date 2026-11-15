// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert store-and-forward forwarder, BRISC (NOC0): walks this directed ring's stream (wr_seq.hpp) in order.
//   INJ:      reads the packet's chunk from this chip's DRAM into CB 0 (RD_DEPTH chunks in flight, one transaction id
//             each; slices stored chunk-contiguous, ws_fab.slice_layout) for the NCRISC to send.
//   FWD/TERM: once the packet is in staging slot k % S (arrival count > k), writes it into its group core's ring slot
//             (the slice's first packet waits until the core freed the slot's previous expert: credit word >= n - 1,
//             fanned out by wr_relay); the slice's last packet then increments the core's arrival word for the slot.
//             The landed count goes to this core's W_LANDED word, which the NCRISC folds into the staging credits.
// Reads still in flight are retired before the walk waits on the ring: the NCRISC sends in stream order, so an
// unpushed chunk would hold back the ring behind it.
// RT: 0 dir, 1 lane, 2 N (ring chips), 3 (unused), 4 slices addr, 5 this core's NoC xy (x | y << 8),
//     6.. hop[NCHIP] (hops from each owner chip to this chip along the ring), then NC targets (c | x << 8 | y << 16)
// CT: 0 NEXP, 1 KT, 2 KTD, 3 DCOLS, 4 CHT, 5 G, 6 GC, 7 NCHIP, 8 NL (lanes), 9 S, 10 wgu_ring, 11 wd_ring, 12 warr,
//     13 words, 14 icred (group slot credit word c), then TensorAccessorArgs(slices)
#include "api/dataflow/dataflow_api.h"
#include "wr_seq.hpp"

void kernel_main() {
    constexpr uint32_t NEXP = get_compile_time_arg_val(0), KT = get_compile_time_arg_val(1);
    constexpr uint32_t KTD = get_compile_time_arg_val(2), DCOLS = get_compile_time_arg_val(3);
    constexpr uint32_t CHT = get_compile_time_arg_val(4), G = get_compile_time_arg_val(5);
    constexpr uint32_t GC = get_compile_time_arg_val(6), NCHIP = get_compile_time_arg_val(7);
    constexpr uint32_t NL = get_compile_time_arg_val(8), S = get_compile_time_arg_val(9);
    constexpr uint32_t wgu_ring = get_compile_time_arg_val(10), wd_ring = get_compile_time_arg_val(11);
    constexpr uint32_t warr = get_compile_time_arg_val(12), words = get_compile_time_arg_val(13);
    constexpr uint32_t icred = get_compile_time_arg_val(14);
    constexpr auto wargs = TensorAccessorArgs<15>();
    constexpr uint32_t WGU = KT * 2, WDN = KTD * DCOLS, WTOT = WGU + WDN;
    static_assert(WGU % CHT == 0 && WTOT % CHT == 0);
    constexpr uint32_t NCH = WTOT / CHT;
    constexpr uint32_t D = RD_DEPTH, SCB = SCB_CHUNKS;  // (D <= SCB, D <= 15)

    const uint32_t dir = get_arg_val<uint32_t>(0), lane = get_arg_val<uint32_t>(1), N = get_arg_val<uint32_t>(2);
    const uint32_t tb = get_tile_size(0);
    const uint32_t CHB = CHT * tb;
    const auto ws = TensorAccessor(wargs, get_arg_val<uint32_t>(4), tb);
    const uint32_t mxy = get_arg_val<uint32_t>(5);
    uint32_t hop[NCHIP];
    for (uint32_t o = 0; o < NCHIP; ++o) {
        hop[o] = get_arg_val<uint32_t>(6 + o);
    }
    constexpr uint32_t TGT = 6 + NCHIP;
    const uint32_t stage = get_write_ptr(1);
    const uint32_t inj = get_write_ptr(0);
    volatile tt_l1_ptr uint32_t* arr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + wr::W_ARR * 16);
    const uint64_t landed_dst = get_noc_addr(mxy & 0xFF, mxy >> 8, words + wr::W_LANDED * 16);

    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    uint32_t t_arr = 0, t_gc = 0, t_land = 0, t_inj = 0;
    const uint32_t tstart = wall();

    uint32_t issued = 0, pushed = 0;
    auto retire = [&]() {
        noc_async_read_barrier_with_trid(1 + pushed % D);
        cb_push_back(0, CHT);
        ++pushed;
    };
    uint32_t k = 0;
    wr::walk<NEXP, NCHIP, G, GC, NL, NCH>(
        hop, dir, lane, N, [&](uint32_t kind, uint32_t e, uint32_t n, uint32_t c, uint32_t u) {
            if (kind == wr::INJ) {
                const uint32_t i0 = wall();
                cb_reserve_back(0, (issued - pushed + 1) * CHT);
                noc_async_read_set_trid(1 + issued % D);
                // chunk q: pages (q / 8) CHT 8 + q % 8 + 8 k (WS_OWN_SLICES: this chip's experts only, e / NCHIP)
#ifdef WS_OWN_SLICES
                const uint32_t q = ((e / NCHIP) * GC + c % GC) * NCH + u;
#else
                const uint32_t q = (e * GC + c % GC) * NCH + u;
#endif
                noc_async_read(ws.get_noc_addr((q / 8) * CHT * 8 + q % 8), inj + (issued % SCB) * CHB, CHB);
                ++issued;
                if (issued - pushed >= D) {
                    retire();
                }
                t_inj += wall() - i0;
                return;
            }
            if (pushed < issued) {
                const uint32_t i0 = wall();
                while (pushed < issued) {
                    retire();
                }
                t_inj += wall() - i0;
            }
            const uint32_t a0 = wall();
            do {
                invalidate_l1_cache();
            } while (*arr < k + 1);
            const uint32_t a1 = wall();
            t_arr += a1 - a0;
            if (u == 0 && n >= 2) {
                volatile tt_l1_ptr uint32_t* cr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(icred + c * 16);
                do {
                    invalidate_l1_cache();
                } while (*cr < n - 1);
            }
            const uint32_t a2 = wall();
            t_gc += a2 - a1;
            const uint32_t t = get_arg_val<uint32_t>(TGT + c);
            const uint32_t tx = (t >> 8) & 0xFF, ty = (t >> 16) & 0xFF;
            const uint32_t u0 = u * CHT;
            const uint32_t dst =
                u0 < WGU ? wgu_ring + ((n & 1) * WGU + u0) * tb : wd_ring + ((n & 1) * WDN + u0 - WGU) * tb;
            noc_async_write(stage + (k % S) * CHB, get_noc_addr(tx, ty, dst), CHB);
            if (u == NCH - 1) {
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(tx, ty, warr + (n & 1) * 16), 1);
            } else {
                noc_async_writes_flushed();
            }
            ++k;
            noc_inline_dw_write(landed_dst, k);
            t_land += wall() - a2;
        });
    while (pushed < issued) {
        retire();
    }
    noc_async_read_set_trid(0);
    noc_async_full_barrier();
    volatile tt_l1_ptr uint32_t* sst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SST_ADDR);
    sst[16] = t_arr;
    sst[17] = t_gc;
    sst[18] = t_land;
    sst[19] = t_inj;
    sst[20] = wall() - tstart;
    sst[21] = k;
    sst[22] = issued;
}
