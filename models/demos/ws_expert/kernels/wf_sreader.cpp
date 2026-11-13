// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert fabric weight landing, sender chip, reader (BRISC, NOC0): for every expert in the global order, for each
// of this connection's target cores of the expert's group (target c: group c / GC, core j = c % GC), reads the core's
// weight slice from this chip's DRAM (gate/up pair j: [KT][2] tiles, then down columns [DCOLS j, DCOLS (j + 1)):
// [KTD][DCOLS]) into CB 0 in chunks of CHT tiles, for the fabric sender (NCRISC).
// RT: 0 wgu addr, 1 wd addr, 2 ntargets, 3.. targets (c | x << 8 | y << 16)
// CT: 0 NEXP, 1 KT, 2 KTD, 3 DCOLS, 4 PAIRS, 5 HT, 6 CHT, 7 G, 8 GC, then TensorAccessorArgs(wgu), (wd)
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NEXP = get_compile_time_arg_val(0), KT = get_compile_time_arg_val(1);
    constexpr uint32_t KTD = get_compile_time_arg_val(2), DCOLS = get_compile_time_arg_val(3);
    constexpr uint32_t PAIRS = get_compile_time_arg_val(4), HT = get_compile_time_arg_val(5);
    constexpr uint32_t CHT = get_compile_time_arg_val(6), G = get_compile_time_arg_val(7);
    constexpr uint32_t GC = get_compile_time_arg_val(8);
    constexpr auto gargs = TensorAccessorArgs<9>();
    constexpr auto dargs = TensorAccessorArgs<gargs.next_compile_time_args_offset()>();
    constexpr uint32_t WGU = KT * 2, WTOT = KT * 2 + KTD * DCOLS;
    static_assert(WGU % CHT == 0 && WTOT % CHT == 0);
    const uint32_t tb = get_tile_size(0);
    const auto gs = TensorAccessor(gargs, get_arg_val<uint32_t>(0), tb);
    const auto ds = TensorAccessor(dargs, get_arg_val<uint32_t>(1), tb);
    const uint32_t nt = get_arg_val<uint32_t>(2);
    for (uint32_t e = 0; e < NEXP; ++e) {
        const uint32_t g = e % G;
        for (uint32_t t = 0; t < nt; ++t) {
            const uint32_t c = get_arg_val<uint32_t>(3 + t) & 0xFF;
            if (c / GC != g) {
                continue;
            }
            const uint32_t j = c % GC;
            for (uint32_t u0 = 0; u0 < WTOT; u0 += CHT) {
                cb_reserve_back(0, CHT);
                uint32_t l1 = get_write_ptr(0);
                for (uint32_t u = u0; u < u0 + CHT; ++u, l1 += tb) {
                    if (u < WGU) {
                        noc_async_read_page((e * KT + u / 2) * 2 * PAIRS + 2 * j + u % 2, gs, l1);
                    } else {
                        const uint32_t v = u - WGU;
                        noc_async_read_page((e * KTD + v / DCOLS) * HT + DCOLS * j + v % DCOLS, ds, l1);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(0, CHT);
            }
        }
    }
}
