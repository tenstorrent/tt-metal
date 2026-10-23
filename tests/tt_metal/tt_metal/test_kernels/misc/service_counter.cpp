// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "internal/firmware_common.h"

void kernel_main() {
    uint32_t stop_addr = get_arg_val<uint32_t>(0);
    uint32_t counter_addr = get_arg_val<uint32_t>(1);
    uint32_t service_done_addr = get_arg_val<uint32_t>(2);

    volatile tt_l1_ptr uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr);
    volatile tt_l1_ptr uint32_t* counter = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counter_addr);
    volatile tt_l1_ptr uint32_t* service_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(service_done_addr);

    while (*stop == 0) {
        (*counter)++;
    }

    // Signal host that the kernel has exited
    *service_done = 1;
}
