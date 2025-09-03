// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

// CT (compile-time) args:
//   none
// RT (runtime) args:
//   0: completion_sem_addr   (u32)  // L1 address of the global semaphore on receiver
//   1: expected_value        (u32)  // e.g. number of pages, or just 1

void kernel_main() {
    size_t idx = 0;
    const uint32_t sem_addr = get_arg_val<uint32_t>(idx++);
    const uint32_t expected_value = get_arg_val<uint32_t>(idx++);

    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    const uint64_t expected_noc = safe_get_noc_addr(my_x[0], my_y[0], sem_addr);

    uint32_t exp_lo = (uint32_t)(expected_noc & 0xffffffffull);
    uint32_t exp_hi = (uint32_t)(expected_noc >> 32);

    DPRINT << "[RX] wait sem=0x" << HEX() << sem_addr << " noc=0x" << exp_hi << "_" << exp_lo << DEC()
           << " expect=" << expected_value << " core=(" << (uint32_t)my_x[0] << "," << (uint32_t)my_y[0] << ")\n";

    noc_semaphore_wait(sem_ptr, expected_value);
    DPRINT << "[RX] done; sem >= " << expected_value << "\n";
}
