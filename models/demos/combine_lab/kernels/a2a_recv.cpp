// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// A2A receiver: waits until every flow into this chip has landed its last row (the counter reaches WANT).
// CT: 0 WANT, 1 the counter address
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t WANT = get_compile_time_arg_val(0), cnt_addr = get_compile_time_arg_val(1);
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cnt_addr);
    while (true) {
        invalidate_l1_cache();
        if (*cnt >= WANT) {
            break;
        }
    }
    *cnt = 0;  // (zero for the next launch: every flow has landed)
}
