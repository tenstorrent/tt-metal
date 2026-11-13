// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert groups, weight injector reader (BRISC, NOC0) - single-chip stand-in for the fabric's weight all-gather:
// for every expert in the global order, for each of this injector's target cores j (j = i + 10 t), reads the core's
// weight slice from DRAM (gate/up pair j, all KT: [KT][2] tiles; down columns [DCOLS j, DCOLS (j + 1)): [KTD][DCOLS])
// into CB 0 in chunks of CH tiles for the writer.
// RT: 0 injector index i, 1 wgu addr, 2 wd addr. CT: 0 NEXP, 1 KT, 2 KTD, 3 DCOLS, 4 PAIRS, 5 HT, 6 CH, 7 NT
// (targets an injector), 8 NINJ, then TensorAccessorArgs(wgu), (wd)
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NEXP = get_compile_time_arg_val(0), KT = get_compile_time_arg_val(1);
    constexpr uint32_t KTD = get_compile_time_arg_val(2), DCOLS = get_compile_time_arg_val(3);
    constexpr uint32_t PAIRS = get_compile_time_arg_val(4), HT = get_compile_time_arg_val(5);
    constexpr uint32_t CH = get_compile_time_arg_val(6), NT = get_compile_time_arg_val(7);
    constexpr uint32_t NINJ = get_compile_time_arg_val(8);
    constexpr auto gargs = TensorAccessorArgs<9>();
    constexpr auto dargs = TensorAccessorArgs<gargs.next_compile_time_args_offset()>();
    constexpr uint32_t WGU = KT * 2, WTOT = KT * 2 + KTD * DCOLS;
    static_assert(WGU % CH == 0 && WTOT % CH == 0);
    const uint32_t i = get_arg_val<uint32_t>(0);
    const uint32_t tb = get_tile_size(0);
    const auto gs = TensorAccessor(gargs, get_arg_val<uint32_t>(1), tb);
    const auto ds = TensorAccessor(dargs, get_arg_val<uint32_t>(2), tb);
    for (uint32_t e = 0; e < NEXP; ++e) {
        for (uint32_t t = 0; t < NT; ++t) {
            const uint32_t j = i + NINJ * t;
            for (uint32_t u0 = 0; u0 < WTOT; u0 += CH) {
                cb_reserve_back(0, CH);
                uint32_t l1 = get_write_ptr(0);
                for (uint32_t u = u0; u < u0 + CH; ++u, l1 += tb) {
                    if (u < WGU) {
                        noc_async_read_page((e * KT + u / 2) * 2 * PAIRS + 2 * j + u % 2, gs, l1);
                    } else {
                        const uint32_t v = u - WGU;
                        noc_async_read_page((e * KTD + v / DCOLS) * HT + DCOLS * j + v % DCOLS, ds, l1);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(0, CH);
            }
        }
    }
}
