// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Lane fabric bench, receiver (a group core's BRISC): waits until the counter of every active direction (fabric atomic
// incs, 16 B a direction) reaches NPKT and stamps each direction's first and last arrival (st[4 + 2 d], st[5 + 2 d]).
// CT: 0 NPKT, 1 counters addr, 2 direction mask; RT: 0 stamps addr
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NPKT = get_compile_time_arg_val(0), cnt_addr = get_compile_time_arg_val(1);
    constexpr uint32_t dmask = get_compile_time_arg_val(2);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cnt_addr);
    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    uint32_t first[2] = {0, 0}, last[2] = {0, 0};
    bool done[2] = {!(dmask & 1), !(dmask & 2)};
    while (!done[0] || !done[1]) {
        invalidate_l1_cache();
        for (uint32_t d = 0; d < 2; ++d) {
            if (done[d]) {
                continue;
            }
            const uint32_t c = cnt[d * 4];
            if (c > 0 && first[d] == 0) {
                first[d] = wall();
            }
            if (c >= NPKT) {
                last[d] = wall();
                done[d] = true;
            }
        }
    }
    for (uint32_t d = 0; d < 2; ++d) {
        st[4 + 2 * d] = first[d];
        st[5 + 2 * d] = last[d];
    }
}
