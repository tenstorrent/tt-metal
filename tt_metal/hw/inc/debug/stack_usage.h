// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "watcher_common.h"

// We don't control the stack size for active erisc, and share the stack with base FW, so don't implement for ERISC.
#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_STACK_USAGE) && !defined(COMPILE_FOR_ERISC)

#define STACK_DIRTY_PATTERN 0xBABABABA

uint32_t get_stack_size() {
#if defined(COMPILE_FOR_BRISC)
    return MEM_BRISC_STACK_SIZE;
#elif defined(COMPILE_FOR_NCRISC)
    return MEM_NCRISC_STACK_SIZE;
#elif defined(COMPILE_FOR_IDLE_ERISC)
    return MEM_IERISC_STACK_SIZE;
#elif COMPILE_FOR_TRISC == 0
    return MEM_TRISC0_STACK_SIZE;
#elif COMPILE_FOR_TRISC == 1
    return MEM_TRISC1_STACK_SIZE;
#elif COMPILE_FOR_TRISC == 2
    return MEM_TRISC2_STACK_SIZE;
#else
    return 0;
#endif
}

uint32_t get_stack_base() {
#if defined(COMPILE_FOR_BRISC)
    return MEM_BRISC_STACK_BASE;
#elif defined(COMPILE_FOR_NCRISC)
    return MEM_NCRISC_STACK_BASE;
#elif defined(COMPILE_FOR_IDLE_ERISC)
    return MEM_IERISC_STACK_BASE;
#elif COMPILE_FOR_TRISC == 0
    return MEM_TRISC0_STACK_BASE;
#elif COMPILE_FOR_TRISC == 1
    return MEM_TRISC1_STACK_BASE;
#elif COMPILE_FOR_TRISC == 2
    return MEM_TRISC2_STACK_BASE;
#else
    return 0;
#endif
}

uint32_t get_dispatch_class() {
#if defined(COMPILE_FOR_BRISC)
    return DISPATCH_CLASS_TENSIX_DM0;
#elif defined(COMPILE_FOR_NCRISC)
    return DISPATCH_CLASS_TENSIX_DM1;
#elif defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC)
    return DISPATCH_CLASS_ETH_DM0;
#else
    return DISPATCH_CLASS_TENSIX_COMPUTE;
#endif
}

void dirty_stack_memory() {
    // Dirty the back 3/4 of the stack
    constexpr uint32_t stack_dirty_fraction_numerator = 3;
    constexpr uint32_t stack_dirty_fraction_denominator = 4;
    uint32_t tt_l1_ptr *stack_ptr = (uint32_t tt_l1_ptr *) get_stack_base();
    uint32_t stack_size = get_stack_size();
    uint32_t stack_max_offset = stack_size / sizeof(uint32_t);
    uint32_t stack_dirty_offset = stack_max_offset * stack_dirty_fraction_numerator / stack_dirty_fraction_denominator;

    // Dirty the back 3/4 of the stack (does rely on the fact that we haven't hit this part of the stack yet).
    for (uint32_t stack_offset = 0; stack_offset < stack_dirty_offset; stack_offset++) {
        stack_ptr[stack_offset] = STACK_DIRTY_PATTERN;
    }
    return;
}

void record_stack_usage() {
    // Write the pause flag for this core into the memory mailbox for host to read.
    debug_stack_usage_t tt_l1_ptr *stack_usage_msg = GET_MAILBOX_ADDRESS_DEV(watcher.stack_usage);
    uint32_t launch_msg_rd_ptr = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
    launch_msg_t tt_l1_ptr *launch_msg = GET_MAILBOX_ADDRESS_DEV(launch[launch_msg_rd_ptr]);
    uint32_t stack_size = get_stack_size();

    uint32_t tt_l1_ptr *stack_ptr = (uint32_t tt_l1_ptr *) get_stack_base();
    for (uint32_t stack_offset = 0; stack_offset < (get_stack_size() / sizeof(uint32_t)); stack_offset++) {
        // If we don't find the dirty pattern, this is the highest the stack has gotten, just store that and return.
        if (stack_ptr[stack_offset] != STACK_DIRTY_PATTERN) {
            // Only update if the stack size used in this kernel is larger than what we've seen before.
            uint16_t stack_usage = stack_size - stack_offset * sizeof(uint32_t);
            if (stack_usage_msg->watcher_kernel_id[debug_get_which_riscv()] == 0 || // No entry recorded
                stack_usage_msg->max_usage[debug_get_which_riscv()] < stack_usage) {
                stack_usage_msg->max_usage[debug_get_which_riscv()] = stack_size - stack_offset * sizeof(uint32_t);
                stack_usage_msg->watcher_kernel_id[debug_get_which_riscv()] =
                    launch_msg->kernel_config.watcher_kernel_ids[get_dispatch_class()];
            }
            return;
        }
    }
}

#define DIRTY_STACK_MEMORY() dirty_stack_memory()
#define RECORD_STACK_USAGE() record_stack_usage()

#else // !WATCHER_ENABLED

#define DIRTY_STACK_MEMORY()
#define RECORD_STACK_USAGE()

#endif // WATCHER_ENABLED
