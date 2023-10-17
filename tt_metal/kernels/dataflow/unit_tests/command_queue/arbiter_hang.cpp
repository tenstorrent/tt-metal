// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    for (uint32_t i = 0; i < 20; i++) {
        uint32_t load = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(400 * 1024);
        uint32_t local_load1 = *reinterpret_cast<volatile uint32_t*>(MEM_LOCAL_BASE);
        uint32_t local_load2 = *reinterpret_cast<volatile uint32_t*>(MEM_LOCAL_BASE);
        uint32_t local_load3 = *reinterpret_cast<volatile uint32_t*>(MEM_LOCAL_BASE);
        uint32_t local_load4 = *reinterpret_cast<volatile uint32_t*>(MEM_LOCAL_BASE);
        uint32_t local_load5 = *reinterpret_cast<volatile uint32_t*>(MEM_LOCAL_BASE);
    }
}
