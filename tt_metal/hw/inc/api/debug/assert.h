// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/debug/watcher_common.h"
#include "internal/hw_thread.h"
#include "risc_common.h"

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ASSERT) && !defined(FORCE_WATCHER_OFF)

//  - for Quasar, multiple DMs and TRISCs share assert_status; only the first to assert records its
//    metadata via a dedicated claim field atomically claimed (amoswap on the cached L1 alias).
//    Writes are flushed to make them visible to host.
inline void assert_and_hang(uint32_t line_num, debug_assert_type_t assert_type = DebugAssertTripped) {
    // Write the line number into the memory mailbox for host to read.
    debug_assert_msg_t tt_l1_ptr* v = GET_MAILBOX_ADDRESS_DEV(watcher.assert_status);
#if defined(ARCH_QUASAR)
    // TODO: Remove this check once mailbox is accessed via cached memory (see dm.cc UNCACHED_MEM_MAILBOX_BASE)
    uintptr_t addr = reinterpret_cast<uintptr_t>(v);
    if (addr >= MEM_L1_UNCACHED_BASE) {
        v = reinterpret_cast<debug_assert_msg_t*>(addr - MEM_L1_UNCACHED_BASE);
    }
    uint32_t old = __atomic_exchange_n(&v->claim, 0xDEADBEEF, __ATOMIC_ACQ_REL);
    if (!old)
#else
    if (v->tripped == DebugAssertOK)
#endif
    {
        v->line_num = line_num;
        v->which = internal_::get_hw_thread_idx();
        if (assert_type == DebugAssertHwFault) {  // only valid on Quasar
            uint64_t mcause;
            uint64_t mtval;
            uint64_t mepc;
            asm volatile("csrr %0, mepc" : "=r"(mepc));
            asm volatile("csrr %0, mcause" : "=r"(mcause));
            asm volatile("csrr %0, mtval" : "=r"(mtval));
            v->line_num = mepc;  // mepc is the instruction address that caused the fault
            v->hw_fault_info = mtval << 32 | (mcause & 0xffffffff);  // mtval is the faulting address or instruction
        }
        v->tripped = assert_type;
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
        // Flush 64B cache line to L1 so host sees all fields via NOC; fence ensures completion
        flush_l2_cache_line(reinterpret_cast<uintptr_t>(v));
#endif
    }

    // Hang, or in the case of erisc, early exit.
#if defined(COMPILE_FOR_ERISC)
    // Update launch msg to show that we've exited. This is required so that the next run doesn't think there's a kernel
    // still running and try to make it exit.
    volatile tt_l1_ptr go_msg_t* go_message_ptr = GET_MAILBOX_ADDRESS_DEV(go_messages[0]);
    go_message_ptr->signal = RUN_MSG_DONE;

    // This exits to base FW
    internal_::disable_erisc_app();
    // Subordinates do not have an erisc exit
#if (defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)) || !defined(ARCH_BLACKHOLE)
    erisc_exit();
#endif
#endif

    while (1) {
        ;
    }
}

// The do... while(0) in this macro allows for it to be called more flexibly, e.g. in an if-else
// without {}s.
#define ASSERT(condition, ...)                        \
    do {                                              \
        if (not(condition))                           \
            assert_and_hang(__LINE__, ##__VA_ARGS__); \
    } while (0)

#define ASSERT_ENABLED 1
#define WATCHER_ASSERT_ENABLED 1
#define LIGHTWEIGHT_ASSERT_ENABLED 0

#else  // !WATCHER_ENABLED

#if defined(LIGHTWEIGHT_KERNEL_ASSERTS)

#define ASSERT(condition, ...)      \
    do {                            \
        if (!(condition)) {         \
            asm volatile("ebreak"); \
        }                           \
    } while (0)

#define ASSERT_ENABLED 1
#define LIGHTWEIGHT_ASSERT_ENABLED 1
#define WATCHER_ASSERT_ENABLED 0

#elif defined(ENABLE_LLK_ASSERT)

#define ASSERT(condition, ...)

#define ASSERT_ENABLED 0
#define LIGHTWEIGHT_ASSERT_ENABLED 0
#define WATCHER_ASSERT_ENABLED 0

#else  // No asserts enabled

#define ASSERT(condition, ...)

#define ASSERT_ENABLED 0
#define WATCHER_ASSERT_ENABLED 0
#define LIGHTWEIGHT_ASSERT_ENABLED 0

#endif  // LIGHTWEIGHT_KERNEL_ASSERTS / ENABLE_LLK_ASSERT

#endif  // WATCHER_ENABLED
