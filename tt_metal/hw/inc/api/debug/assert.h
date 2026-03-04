// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/debug/watcher_common.h"
#include "internal/hw_thread.h"

#if defined(ARCH_QUASAR)
#include "internal/tt-2xx/quasar/overlay/overlay_addresses.h"
#endif

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ASSERT) && !defined(FORCE_WATCHER_OFF)

// assert_and_hang captures its return address (the call site inside the ASSERT macro) via
// __builtin_return_address(0) and stores it in the mailbox pc field.  The host resolves
// file, line, function, and message from that PC using DWARF — no file hash needed.
// For Quasar, multiple DMs share assert_status; CAS is used for thread safety; address is
// remapped from uncached to cached L1 (LR/SC requires cache coherence), then flushed to host.
inline void assert_and_hang(debug_assert_type_t assert_type = DebugAssertTripped) {
    debug_assert_msg_t tt_l1_ptr* v = GET_MAILBOX_ADDRESS_DEV(watcher.assert_status);
#if defined(ARCH_QUASAR)
    // TODO: Remove this check once mailbox is accessed via cached memory (see dm.cc UNCACHED_MEM_MAILBOX_BASE)
    uintptr_t addr = reinterpret_cast<uintptr_t>(v);
    if (addr >= MEM_L1_UNCACHED_BASE) {
        v = reinterpret_cast<debug_assert_msg_t*>(addr - MEM_L1_UNCACHED_BASE);
    }
    uint8_t expected = DebugAssertOK;
    if (__atomic_compare_exchange_n(
            &v->tripped, &expected, DebugAssertWriteInProgress, false, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED))
#else
    if (v->tripped == DebugAssertOK)
#endif
    {
        v->pc = (uint32_t)__builtin_return_address(0);
        v->which = internal_::get_hw_thread_idx();
        if (assert_type == DebugAssertHwFault) {  // only valid on Quasar
            uint64_t mcause;
            uint64_t mtval;
            uint64_t mepc;
            asm volatile("csrr %0, mepc" : "=r"(mepc));
            asm volatile("csrr %0, mcause" : "=r"(mcause));
            asm volatile("csrr %0, mtval" : "=r"(mtval));
            v->pc = (uint32_t)mepc;  // mepc is the instruction address that caused the fault
            v->hw_fault_info = mtval << 32 | (mcause & 0xffffffff);
        }
        v->tripped = assert_type;
#if defined(ARCH_QUASAR)
        // Flush 64B cache line to L1 so host sees all fields via NOC; fence ensures completion
        // TODO: Replace with flush_l2_cache_line() once available
        volatile uint64_t* flush_reg = reinterpret_cast<volatile uint64_t*>(L2_FLUSH_ADDR);
        *flush_reg = reinterpret_cast<uintptr_t>(v);
        asm volatile("fence" ::: "memory");
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

// Overload for ASSERT(condition, "message") — the string is developer documentation only;
// the host reads it from source at the resolved file:line, nothing needs to be stored.
inline void assert_and_hang(const char*) {
    assert_and_hang();
}

// The do...while(0) allows flexible use in if-else without braces.
// Optional second argument: enum assert type (e.g. DebugAssertRtaOutOfBounds)
//                        or string message (developer documentation, ignored at runtime).
#define ASSERT(condition, ...) \
    do { \
        if (not(condition)) \
            assert_and_hang(##__VA_ARGS__); \
    } while (0)

#define ASSERT_ENABLED 1
#define WATCHER_ASSERT_ENABLED 1
#define LIGHTWEIGHT_ASSERT_ENABLED 0

#else  // !WATCHER_ENABLED

#if defined(LIGHTWEIGHT_KERNEL_ASSERTS)

#define ASSERT(condition, ...) \
    do {                       \
        if (!(condition))      \
            while (true);      \
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
