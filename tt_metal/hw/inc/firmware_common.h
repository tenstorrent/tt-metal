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

extern void (*__init_array_start[])();
extern void (*__init_array_end[])();

extern void kernel_init(uint32_t kernel_init);
extern void kernel_launch(uint32_t kernel_base_addr);

inline void l1_to_local_mem_copy(uint32_t* dst, uint32_t tt_l1_ptr* src, int32_t len) {
#pragma GCC unroll 0
    while (len >= 3) {
        auto v0 = src[0], v1 = src[1], v2 = src[2];
        // 1) Make sure the optimizer does not think this is memcpy by
        // hiding the pointer bookkeeping in an asm.
        // 2) The scheduler doesn't know the above loads have 6 cycle
        // latency. We emit the 3 bookkeeping adds as a single block
        // in the load shadow before the stores. The optimizer will
        // not be able to move these.
        // 3) We don't need early clobbers here because of the +r
        // constraint -- early clobbers would pessimize.
        asm inline(
            "addi %0,%0,3*%3\n\t"
            "addi %1,%1,3*%3\n\t"
            "addi %2,%2,-3"
            : "+r"(src), "+r"(dst), "+r"(len)
            : "i"(sizeof(v0)));
        dst[-3] = v0, dst[-2] = v1, dst[-1] = v2;
    }
    // There are 0, 1 or 2 words of residue. This is smaller than a loop.
    // We get smaller code layout by expecting the conditions to be true.
    if (__builtin_expect(len >= 1, true)) {
        dst[0] = src[0];
        if (__builtin_expect(len >= 2, true)) {
            dst[1] = src[1];
        }
    }
}

inline void do_crt1(uint32_t tt_l1_ptr* data_image) {
    // Clear bss.
    extern uint32_t __ldm_bss_start[];
    extern uint32_t __ldm_bss_end[];
    wzerorange(__ldm_bss_start, __ldm_bss_end);

    // Copy initialized data.
    extern uint32_t __ldm_data_start[];
    extern uint32_t __ldm_data_end[];
    l1_to_local_mem_copy(__ldm_data_start, data_image, __ldm_data_end - __ldm_data_start);

    for (void (**fptr)() = __init_array_start; fptr < __init_array_end; fptr++) {
        (**fptr)();
    }
}

FORCE_INLINE
uint32_t firmware_config_init(
    tt_l1_ptr mailboxes_t* const mailboxes, uint32_t core_type_index, uint32_t dispatch_class) {
    extern uint32_t tt_l1_ptr* rta_l1_base;
    extern uint32_t tt_l1_ptr* crta_l1_base;
    extern uint32_t tt_l1_ptr* sem_l1_base[ProgrammableCoreType::COUNT];

    // TODO: check the asm for this loop to be sure loads are scheduled ok
    uint32_t kernel_config_base[ProgrammableCoreType::COUNT];
    launch_msg_t* launch_msg_address = &(mailboxes->launch[mailboxes->launch_msg_rd_ptr]);
#pragma GCC unroll ProgrammableCoreType::COUNT
    for (uint32_t index = 0; index < ProgrammableCoreType::COUNT; index++) {
        kernel_config_base[index] = launch_msg_address->kernel_config.kernel_config_base[index];
        sem_l1_base[index] =
            (uint32_t tt_l1_ptr*)(kernel_config_base[index] + launch_msg_address->kernel_config.sem_offset[index]);
    }
    rta_l1_base = (uint32_t tt_l1_ptr*)(kernel_config_base[core_type_index] +
                                        launch_msg_address->kernel_config.rta_offset[dispatch_class].rta_offset);
    crta_l1_base = (uint32_t tt_l1_ptr*)(kernel_config_base[core_type_index] +
                                         launch_msg_address->kernel_config.rta_offset[dispatch_class].crta_offset);

    return kernel_config_base[core_type_index];
}
