// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/assert.h"
#include "debug/dprint.h"
#include "dataflow_api.h"
#include <array>


void kernel_main() {
    std::array<uint32_t, 8> channels_addrs;
    std::array<uint32_t, 8> channels_sem_addrs;
    uint32_t arg_idx = 0;
    volatile uint32_t* sem0 = reinterpret_cast<volatile uint32_t*>(get_semaphore(get_arg_val<uint32_t>(arg_idx++)));
    volatile uint32_t* sem1 = reinterpret_cast<volatile uint32_t*>(get_semaphore(get_arg_val<uint32_t>(arg_idx++)));
    uint32_t core0_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t core0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t core0_start_sem = get_arg_val<uint32_t>(arg_idx++);

    noc_semaphore_wait(sem0, 1);
    noc_semaphore_wait(sem1, 1);

    uint64_t core0_start_sem_noc_addr = get_noc_addr(core0_x, core0_y, core0_start_sem);

    noc_semaphore_inc(core0_start_sem_noc_addr, 1);
}
