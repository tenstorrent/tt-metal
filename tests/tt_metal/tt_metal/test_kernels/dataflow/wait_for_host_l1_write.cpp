// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/debug/dprint.h"

void kernel_main() {
    set_l1_data_cache<true>();

    uint32_t release_addr = get_arg_val<uint32_t>(0);
    uint32_t release_value = get_arg_val<uint32_t>(1);
    uint32_t started_addr = get_arg_val<uint32_t>(2);
    uint32_t started_value = get_arg_val<uint32_t>(3);

    volatile tt_l1_ptr uint32_t* release_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(release_addr);
    volatile tt_l1_ptr uint32_t* started_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(started_addr);

    started_ptr[0] = started_value;

    while (release_ptr[0] != release_value) {
        invalidate_l1_cache();
    }

    set_l1_data_cache<false>();
}
