// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "watcher_common.h"
#include "dev_mem_map.h"

// We don't control the stack size for active erisc, and share the stack with base FW, so don't implement for ERISC.
#if defined(WATCHER_ENABLED) && \
    !(defined(WATCHER_DISABLE_STACK_USAGE) || defined(COMPILE_FOR_ERISC) || defined(FORCE_WATCHER_OFF))

#if defined(KERNEL_BUILD)

constexpr uint32_t stack_usage_pattern = 0xBABABABA;

// Helper to resolve the stack base without triggering a global constructor
static inline uint32_t* get_stack_base() {
#if defined(ARCH_QUASAR)
    extern thread_local uint32_t __stack_base_lwm[];
    extern char __stack_base_offset;
    uint32_t* __stack_base = &__stack_base_lwm[uintptr_t(&__stack_base_offset)];
    return __stack_base;
#else
    extern uint32_t __stack_base[];
    return __stack_base;
#endif
}

static inline void mark_stack_usage() {
    uint32_t* __stack_base = get_stack_base();
    uint32_t tt_l1_ptr *ptr;
    asm ("mv %0,sp" : "=r"(ptr));

    while (ptr != __stack_base) {
        *--ptr = stack_usage_pattern;
    }
}

// Returns unused stack + 1. (0 means unknown.)
static inline uint32_t measure_stack_usage() {
    uint32_t* __stack_base = get_stack_base();
    uint32_t tt_l1_ptr* stack_ptr = __stack_base;
    // We don't need to check size here, as we know we'll hit a
    // non-dirty value at some point (a set of return addresses).
    while (*stack_ptr == stack_usage_pattern) {
        stack_ptr++;
    }
    uint32_t stack_free = (uint32_t)((uintptr_t)stack_ptr - (uintptr_t)&__stack_base[0]);
    return stack_free + 1;
}

#else   // !KERNEL_BUILD

// stack_free is offset by 1
static inline void record_stack_usage(uint32_t stack_free) {
    if (!stack_free) {
        // not computed
        return;
    }

    std::uint64_t cpu_index = 0;
    // TODO: This logic should be moved to an internal get_cpu_idx() API.
    // Deferred to a follow-up PR
#if defined(ARCH_QUASAR)
#if defined(COMPILE_FOR_TRISC)
    std::uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();
    std::uint32_t trisc_id = ckernel::csr_read<ckernel::CSR::TRISC_ID>();
    cpu_index = NUM_DM_CORES + NUM_TRISC_CORES * neo_id + trisc_id;  // after 8 DM cores
#else
    asm volatile("csrr %0, mhartid" : "=r"(cpu_index));
#endif
#else
    cpu_index = PROCESSOR_INDEX;
#endif
    auto tt_l1_ptr* usage = &GET_MAILBOX_ADDRESS_DEV(watcher.stack_usage)->cpu[cpu_index];
    // min_free is initialized to zero, which we want to compare as
    // least noteworthy, and an offset free stack of one as the most
    // noteworthy. Decrement the former, so zero wraps around before
    // checking. (The cast to uint32_t isn't necessary, but conveys
    // meaning.)
    if (uint32_t(usage->min_free) - 1 >= stack_free) {
        usage->min_free = stack_free;
        unsigned launch_idx = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
        launch_msg_t tt_l1_ptr* launch_msg = GET_MAILBOX_ADDRESS_DEV(launch[launch_idx]);
        usage->watcher_kernel_id = launch_msg->kernel_config.watcher_kernel_ids[cpu_index];
    }
}
#endif  // KERNEL_BUILD

#else  // !WATCHER_ENABLED

static inline void mark_stack_usage() {}
static inline uint32_t measure_stack_usage() { return 0; }
static inline void record_stack_usage(uint32_t) {}

#endif  // WATCHER_ENABLED
