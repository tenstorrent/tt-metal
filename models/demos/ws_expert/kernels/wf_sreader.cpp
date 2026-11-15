// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert fabric weight landing, sender chip, reader (BRISC, NOC0): for every expert in the global order, for each
// of this connection's target cores of the expert's group (target c: group c / GC, core j = c % GC), reads the core's
// weight slice (gate/up pair j: [KT][2] tiles, then down columns [DCOLS j, DCOLS (j + 1)): [KTD][DCOLS]) from this
// chip's DRAM into CB 0 (SCB chunks of CHT tiles) for the fabric sender (NCRISC), RD_DEPTH chunks in flight (one
// transaction id each). The slices are stored chunk-contiguous (ws_fab.slice_layout): chunk q = (e GC + j) NCH + u
// is one read of CHT tiles from bank q % 8 (576 B page reads were issue-bound at ~10 GB/s a sender).
// RT: 0 slices addr, 1 own (reads experts e % NOWN == own), 2 ntargets, 3.. targets (c | x << 8 | y << 16), then with
//     WS_ITEMS: the number of owned slices, the address of the table holding the list of them (c | e << 8, in order;
//     read instead of own's experts), its page and the page bytes
// WS_OWN_SLICES: the slices tensor holds only the slices this chip owns: slice j of expert e at local index
//     (e / NOWN) GC + j
// CT: 0 NEXP, 1 KT, 2 KTD, 3 DCOLS, 4 PAIRS, 5 HT, 6 CHT, 7 G, 8 GC, then TensorAccessorArgs(slices), with WS_ITEMS
//     TensorAccessorArgs(table)
#include "api/dataflow/dataflow_api.h"

#ifndef NOWN
#define NOWN 1  // one sender chip set reads every expert (step 1); step 2 passes the owner count
#endif

void kernel_main() {
    constexpr uint32_t NEXP = get_compile_time_arg_val(0), KT = get_compile_time_arg_val(1);
    constexpr uint32_t KTD = get_compile_time_arg_val(2), DCOLS = get_compile_time_arg_val(3);
    constexpr uint32_t PAIRS = get_compile_time_arg_val(4), HT = get_compile_time_arg_val(5);
    constexpr uint32_t CHT = get_compile_time_arg_val(6), G = get_compile_time_arg_val(7);
    constexpr uint32_t GC = get_compile_time_arg_val(8);
    constexpr uint32_t SCB = SCB_CHUNKS;
    constexpr auto wargs = TensorAccessorArgs<9>();
    constexpr uint32_t WGU = KT * 2, WTOT_TILES = KT * 2 + KTD * DCOLS;
    static_assert(WGU % CHT == 0 && WTOT_TILES % CHT == 0);
    constexpr uint32_t NCH = WTOT_TILES / CHT;  // chunks a slice
    const uint32_t tb = get_tile_size(0);
    const auto ws = TensorAccessor(wargs, get_arg_val<uint32_t>(0), tb);
    const uint32_t nt = get_arg_val<uint32_t>(2);
    // up to D chunks in flight (transaction id 1 + i % D each), pushed in order as they complete
    constexpr uint32_t D = RD_DEPTH;  // (<= SCB, <= 15)
    const uint32_t CHB = CHT * tb;
    const uint32_t base = get_write_ptr(0);
    uint32_t issued = 0, pushed = 0;
    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    uint32_t t_issue = 0, t_bar = 0, t_res = 0;
    const uint32_t tstart = wall();
    auto retire = [&]() {
        const uint32_t b0 = wall();
        noc_async_read_barrier_with_trid(1 + pushed % D);
        t_bar += wall() - b0;
        cb_push_back(0, CHT);
        ++pushed;
    };
    auto read_slice = [&](uint32_t e, uint32_t j) {
        for (uint32_t u = 0; u < NCH; ++u) {
            const uint32_t r0 = wall();
            cb_reserve_back(0, (issued - pushed + 1) * CHT);
            const uint32_t i0 = wall();
            t_res += i0 - r0;
            uint32_t l1 = base + (issued % SCB) * CHB;
            noc_async_read_set_trid(1 + issued % D);
            // chunk q's CHT tiles are pages (q / 8) CHT 8 + q % 8 + 8 k: contiguous in one bank
#ifdef WS_OWN_SLICES
            const uint32_t q = ((e / NOWN) * GC + j) * NCH + u;  // only the slices this chip owns (local index e / NOWN)
#else
            const uint32_t q = (e * GC + j) * NCH + u;
#endif
            noc_async_read(ws.get_noc_addr((q / 8) * CHT * 8 + q % 8), l1, CHB);
            ++issued;
            t_issue += wall() - i0;
            if (issued - pushed >= D) {
                retire();
            }
        }
    };
#ifdef WS_ITEMS
    // the host's list of owned slices (c | e << 8), in the order the sender lands them: one page of a DRAM table,
    // loaded into CB 7 for the sender (NCRISC) as well
    constexpr auto iargs = TensorAccessorArgs<wargs.next_compile_time_args_offset()>();
    const uint32_t ni = get_arg_val<uint32_t>(3 + nt), ipage_bytes = get_arg_val<uint32_t>(6 + nt);
    const auto ia = TensorAccessor(iargs, get_arg_val<uint32_t>(4 + nt), ipage_bytes);
    cb_reserve_back(7, 1);
    const uint32_t il1 = get_write_ptr(7);
    noc_async_read(ia.get_noc_addr(get_arg_val<uint32_t>(5 + nt)), il1, ipage_bytes);
    noc_async_read_barrier();
    cb_push_back(7, 1);
    volatile tt_l1_ptr uint32_t* items = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(il1);
    for (uint32_t i = 0; i < ni; ++i) {
        const uint32_t it = items[i];
        read_slice(it >> 8, (it & 0xFF) % GC);
    }
#else
    const uint32_t own = get_arg_val<uint32_t>(1);  // this chip's experts: e % NOWN == own
    for (uint32_t e = 0; e < NEXP; ++e) {
        if (e % NOWN != own) {
            continue;
        }
        const uint32_t g = e % G;
        for (uint32_t t = 0; t < nt; ++t) {
            const uint32_t c = get_arg_val<uint32_t>(3 + t) & 0xFF;
            if (c / GC == g) {
                read_slice(e, c % GC);
            }
        }
    }
#endif
    while (pushed < issued) {
        retire();
    }
    noc_async_read_set_trid(0);
    volatile tt_l1_ptr uint32_t* sst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SST_ADDR);
    sst[4] = t_res;
    sst[5] = t_issue;
    sst[6] = t_bar;
    sst[7] = wall() - tstart;
}
