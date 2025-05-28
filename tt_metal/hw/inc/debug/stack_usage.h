// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "watcher_common.h"

// We don't control the stack size for active erisc, and share the stack with base FW, so don't implement for ERISC.
#if defined(WATCHER_ENABLED) && !(defined(WATCHER_DISABLE_STACK_USAGE) || defined(COMPILE_FOR_ERISC) || \
                                  defined(FORCE_WATCHER_OFF))

#define STACK_DIRTY_PATTERN 0xBABABABA

static inline uint32_t get_stack_base() {
#if defined(COMPILE_FOR_BRISC)
    return MEM_BRISC_STACK_TOP - MEM_BRISC_STACK_SIZE;
#elif defined(COMPILE_FOR_NCRISC)
    return MEM_NCRISC_STACK_TOP - MEM_NCRISC_STACK_SIZE;
#elif defined(COMPILE_FOR_IDLE_ERISC)
#if COMPILE_FOR_IDLE_ERISC == 0
    return MEM_IERISC_STACK_TOP - MEM_IERISC_STACK_SIZE;
#elif COMPILE_FOR_IDLE_ERISC == 1
    return MEM_SUBORDINATE_IERISC_STACK_TOP - MEM_SUBORDINATE_IERISC_STACK_SIZE;
#else
#error "idle erisc get_stack_base unknown"
#endif
#elif defined(COMPILE_FOR_TRISC)
#if COMPILE_FOR_TRISC == 0
    return MEM_TRISC0_STACK_TOP - MEM_TRISC0_STACK_SIZE;
#elif COMPILE_FOR_TRISC == 1
    return MEM_TRISC1_STACK_TOP - MEM_TRISC1_STACK_SIZE;
#elif COMPILE_FOR_TRISC == 2
    return MEM_TRISC2_STACK_TOP - MEM_TRISC2_STACK_SIZE;
#else
#error "trisc get_stack_base unknown"
#endif
#else
#error "get_stack_base unknown"
#endif
}

static inline uint32_t get_dispatch_class() {
#if defined(COMPILE_FOR_BRISC)
    return DISPATCH_CLASS_TENSIX_DM0;
#elif defined(COMPILE_FOR_NCRISC)
    return DISPATCH_CLASS_TENSIX_DM1;
#elif defined(COMPILE_FOR_ERISC)
    return DISPATCH_CLASS_ETH_DM0;
#elif defined(COMPILE_FOR_IDLE_ERISC)
    return COMPILE_FOR_IDLE_ERISC == 0 ? DISPATCH_CLASS_ETH_DM0
        : DISPATCH_CLASS_ETH_DM1;
#elif defined(COMPILE_FOR_TRISC)
    return DISPATCH_CLASS_TENSIX_COMPUTE;
#else
#error "dispatch class not defined"
#endif
}

void dirty_stack_memory() {
    uint32_t *base = (uint32_t tt_l1_ptr *)get_stack_base();
    uint32_t tt_l1_ptr *ptr;
    asm ("mv %0,sp" : "=r"(ptr));

    while (ptr != base)
        *--ptr = STACK_DIRTY_PATTERN;
}

void record_stack_usage() {
    uint32_t *base = (uint32_t tt_l1_ptr *)get_stack_base();

    uint32_t tt_l1_ptr* stack_ptr = base;
    // We don't need to check size here, as we know we'll hit a
    // non-dirty value at some point (a set of return addresses).
    while (*stack_ptr == STACK_DIRTY_PATTERN)
        stack_ptr++;
    uint32_t stack_free = (stack_ptr - base) * sizeof (uint32_t);
    unsigned idx = debug_get_which_riscv();
    debug_stack_usage_t::usage_t tt_l1_ptr* usage = &GET_MAILBOX_ADDRESS_DEV(watcher.stack_usage)->cpu[idx];
    // min_free is initialized to zero, which we want to compare as
    // least noteworthy, and a free stack of zero as the most
    // noteworthy. Decrement the former, so zero wraps around before
    // checking. (The cast to uint32_t isn't necessary, but conveys
    // meaning.)
    if (uint32_t(usage->min_free) - 1 > stack_free) {
        usage->min_free = stack_free + 1;
        unsigned launch_idx = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
        launch_msg_t tt_l1_ptr* launch_msg = GET_MAILBOX_ADDRESS_DEV(launch[launch_idx]);
        usage->watcher_kernel_id = launch_msg->kernel_config.watcher_kernel_ids[get_dispatch_class()];
    }
}

#define DIRTY_STACK_MEMORY() dirty_stack_memory()
#define RECORD_STACK_USAGE() record_stack_usage()

#else  // !WATCHER_ENABLED

#define DIRTY_STACK_MEMORY()
#define RECORD_STACK_USAGE()

#endif  // WATCHER_ENABLED
