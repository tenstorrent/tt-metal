// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "core_config.h"
#include "ckernel_globals.h"
#include "tensix_functions.h"
#include "risc_attribs.h"
#include "compile_time_args.h"
#include "dev_mem_map.h"
#include "hostdevcommon/kernel_structs.h"
#include "dev_msgs.h"

extern uint32_t __ldm_bss_start[];
extern uint32_t __ldm_bss_end[];
extern uint32_t __ldm_data_start[];
extern uint32_t __ldm_data_end[];
extern void (* __init_array_start[])();
extern void (* __init_array_end[])();

extern void kernel_init();
extern void kernel_launch();

inline void l1_to_local_mem_copy(uint32_t *local_mem_addr, uint32_t tt_l1_ptr *l1_addr, int32_t len) {
    // Cover L1 load latency of 6 cycles for the bulk of the copy
    int32_t n = 0;
    while (n < len - 5) {
        uint32_t v0 = l1_addr[n + 0];
        uint32_t v1 = l1_addr[n + 1];
        uint32_t v2 = l1_addr[n + 2];
        uint32_t v3 = l1_addr[n + 3];
        uint32_t v4 = l1_addr[n + 4];
        uint32_t v5 = l1_addr[n + 5];
        local_mem_addr[n + 0] = v0;
        local_mem_addr[n + 1] = v1;
        local_mem_addr[n + 2] = v2;
        local_mem_addr[n + 3] = v3;
        local_mem_addr[n + 4] = v4;
        local_mem_addr[n + 5] = v5;
        n += 6;
    }
    // Could optimize this further (eg, loop of 2 or 4), probably not worth it
    while (n < len) {
        local_mem_addr[n] = l1_addr[n];
        n++;
    }
}

inline void firmware_kernel_common_init(void *init_local_l1_base) {

    // Handle stuff typically done in crt0 in asm.  Easier to do in C
    wzerorange(__ldm_bss_start, __ldm_bss_end);

    int32_t num_words = ((uint)__ldm_data_end - (uint)__ldm_data_start) >> 2;
    l1_to_local_mem_copy((uint32_t *)__ldm_data_start, (uint32_t *)((uint8_t *)init_local_l1_base), num_words);

    for (void (** fptr)() = __init_array_start; fptr < __init_array_end; fptr++) {
        (**fptr)();
    }
}
FORCE_INLINE
uint32_t firmware_config_init(tt_l1_ptr mailboxes_t* const mailboxes, uint32_t core_type_index, uint32_t dispatch_class) {

    extern uint32_t tt_l1_ptr *rta_l1_base;
    extern uint32_t tt_l1_ptr *crta_l1_base;
    extern uint32_t tt_l1_ptr *sem_l1_base[ProgrammableCoreType::COUNT];

    // TODO: check the asm for this loop to be sure loads are scheduled ok
    uint32_t kernel_config_base[ProgrammableCoreType::COUNT];
#pragma GCC unroll ProgrammableCoreType::COUNT
    for (uint32_t index = 0; index < ProgrammableCoreType::COUNT; index++) {
        kernel_config_base[index] =
            mailboxes->launch.kernel_config.kernel_config_base[index];
        sem_l1_base[index] = (uint32_t tt_l1_ptr *)(kernel_config_base[index] +
            mailboxes->launch.kernel_config.sem_offset[index]);
    }
    rta_l1_base = (uint32_t tt_l1_ptr *)(kernel_config_base[core_type_index] +
        mailboxes->launch.kernel_config.mem_map[dispatch_class].rta_offset);
    crta_l1_base = (uint32_t tt_l1_ptr *)(kernel_config_base[core_type_index] +
        mailboxes->launch.kernel_config.mem_map[dispatch_class].crta_offset);

    return kernel_config_base[core_type_index];
}
