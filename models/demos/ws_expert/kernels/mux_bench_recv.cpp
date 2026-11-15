// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fabric mux throughput bench, receiver: waits until its counter (fabric atomic incs) reaches NPKT and stamps the first
// and the last arrival.
// CT: 0 NPKT, 1 counter addr; RT: 0 stamps addr
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NPKT = get_compile_time_arg_val(0), cnt_addr = get_compile_time_arg_val(1);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cnt_addr);
    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    uint32_t t0 = 0;
    while (true) {
        invalidate_l1_cache();
        const uint32_t c = *cnt;
        if (c > 0 && t0 == 0) {
            t0 = wall();
        }
        if (c >= NPKT) {
            break;
        }
    }
    st[2] = t0;
    st[3] = wall();
}
