// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fabric link microbench, receiver: waits for its counter (the sender's fused write + atomic incs) to reach WANT and
// stamps the wall clock when it first becomes non-zero (the first packet landed) and when it reaches WANT (the last).
// CT: 0 WANT, 1 counter (L1); RT: 0 stamps (L1)
#include "api/dataflow/dataflow_api.h"

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void kernel_main() {
    constexpr uint32_t WANT = get_compile_time_arg_val(0), cnt_addr = get_compile_time_arg_val(1);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cnt_addr);
    uint32_t t0 = 0;
    while (true) {
        invalidate_l1_cache();
        const uint32_t c = *cnt;
        if (c > 0 && t0 == 0) {
            t0 = wall();
        }
        if (c >= WANT) {
            break;
        }
    }
    st[2] = t0;
    st[3] = wall();
}
