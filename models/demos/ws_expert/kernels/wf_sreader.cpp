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
// RT: 0 slices addr, 1 (unused), 2 ntargets, 3.. targets (c | x << 8 | y << 16)
// CT: 0 NEXP, 1 KT, 2 KTD, 3 DCOLS, 4 PAIRS, 5 HT, 6 CHT, 7 G, 8 GC, then TensorAccessorArgs(slices)
#include "api/dataflow/dataflow_api.h"

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
    for (uint32_t e = 0; e < NEXP; ++e) {
        const uint32_t g = e % G;
        for (uint32_t t = 0; t < nt; ++t) {
            const uint32_t c = get_arg_val<uint32_t>(3 + t) & 0xFF;
            if (c / GC != g) {
                continue;
            }
            const uint32_t j = c % GC;
            for (uint32_t u = 0; u < NCH; ++u) {
                const uint32_t r0 = wall();
                cb_reserve_back(0, (issued - pushed + 1) * CHT);
                const uint32_t i0 = wall();
                t_res += i0 - r0;
                uint32_t l1 = base + (issued % SCB) * CHB;
                noc_async_read_set_trid(1 + issued % D);
                // chunk q's CHT tiles are pages (q / 8) CHT 8 + q % 8 + 8 k: contiguous in one bank
                const uint32_t q = (e * GC + j) * NCH + u;
                noc_async_read(ws.get_noc_addr((q / 8) * CHT * 8 + q % 8), l1, CHB);
                ++issued;
                t_issue += wall() - i0;
                if (issued - pushed >= D) {
                    retire();
                }
            }
        }
    }
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
