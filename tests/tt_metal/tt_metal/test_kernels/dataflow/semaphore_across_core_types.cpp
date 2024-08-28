// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/dprint.h"

void kernel_main() {
    constexpr ProgrammableCoreType eth_core_type = static_cast<ProgrammableCoreType>(get_compile_time_arg_val(0));

    uint32_t other_noc_xy = get_arg_val<uint32_t>(0);
    uint32_t my_sem_id = get_arg_val<uint32_t>(1);
    uint32_t other_sem_id = get_arg_val<uint32_t>(2);
    uint32_t sem_init_value = get_arg_val<uint32_t>(3);

#if defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
    volatile tt_l1_ptr uint32_t* my_sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<eth_core_type>(my_sem_id));
    volatile tt_l1_ptr uint32_t* other_sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(other_sem_id));
#endif
#ifdef COMPILE_FOR_BRISC
    volatile tt_l1_ptr uint32_t* my_sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(my_sem_id));
    volatile tt_l1_ptr uint32_t* other_sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore<eth_core_type>(other_sem_id));
#endif

    uint64_t dst_addr = get_noc_addr_helper(other_noc_xy, (uint32_t)other_sem_addr);
    noc_semaphore_inc(dst_addr, 1);

    // Spin until other core updates the local semaphore confirming the addresses are correct
    while (*my_sem_addr == sem_init_value);
}
