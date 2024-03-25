// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel_globals.h"
#include "tensix_functions.h"
#include "risc_attribs.h"
#include "compile_time_args.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "hostdevcommon/kernel_structs.h"

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
    uint32_t offset = (uint32_t)__ldm_data_start - MEM_LOCAL_BASE;
    l1_to_local_mem_copy((uint32_t *)__ldm_data_start, (uint32_t *)((uint8_t *)init_local_l1_base + offset), num_words);

    for (void (** fptr)() = __init_array_start; fptr < __init_array_end; fptr++) {
        (**fptr)();
    }

    // Make sure DBG_FEATURE_DISABLE register is cleared before every kernel is executed
    memory_write(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 0);

}
