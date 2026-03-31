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

// For Quasar, multiple DMs share assert_status; CAS is used for thread safety; address is
// remapped from uncached to cached L1 (LR/SC requires cache coherence), then flushed to host.
inline void assert_and_hang(uint32_t line_num, uint16_t file_id, debug_assert_type_t assert_type = DebugAssertTripped) {
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
        v->line_num = line_num;
        v->file_id = file_id;
        v->which = internal_::get_hw_thread_idx();
        if (assert_type == DebugAssertHwFault) {  // only valid on Quasar
            uint64_t mcause;
            uint64_t mtval;
            uint64_t mepc;
            asm volatile("csrr %0, mepc" : "=r"(mepc));
            asm volatile("csrr %0, mcause" : "=r"(mcause));
            asm volatile("csrr %0, mtval" : "=r"(mtval));
            v->line_num = mepc;  // mepc is the faulting instruction address
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
    internal_::disable_erisc_app();
#if (defined(COMPILE_FOR_AERISC) && (PHYSICAL_AERISC_ID == 0)) || !defined(ARCH_BLACKHOLE)
    erisc_exit();
#endif
#endif

    while (1) {
        ;
    }
}

// Backward-compatible overload — existing callers that only pass assert_type get file_id=0.
inline void assert_and_hang(uint32_t line_num, debug_assert_type_t assert_type = DebugAssertTripped) {
    assert_and_hang(line_num, 0, assert_type);
}
// Overload so the discarded else-branch of _ASSERT_EXTRA type-checks in template bodies
// (GCC -Wtemplate-body checks both branches even when if constexpr discards one).
inline void assert_and_hang(uint32_t line_num, uint16_t file_id, const char*) { assert_and_hang(line_num, file_id); }

// ASSERT(condition)                — hang with DebugAssertTripped
// ASSERT(condition, "message")     — hang; message stored in file-only ELF section (zero L1 cost)
// ASSERT(condition, type)          — hang with a specific debug_assert_type_t
//
// ASSERT message section entry layout (packed, variable-length):
//   [uint16_t file_id][uint16_t line_num][char msg[sizeof(literal)]]

// Implementation detail — do not call directly.
// Single extra arg: string literal → embed in ELF section and hang;
//                   enum type      → hang with that specific assert type.
#define _ASSERT_EXTRA(condition, extra)                                              \
    do {                                                                             \
        if (not(condition)) {                                                        \
            if constexpr (__is_same(__typeof__(extra), const char[sizeof(extra)])) { \
                static const struct __attribute__((packed)) {                        \
                    uint16_t file_id;                                                \
                    uint16_t line_num;                                               \
                    char msg_data[sizeof(extra)];                                    \
                } _e __attribute__((used, section(".debug_assert_msgs"))) = {        \
                    debug_file_hash(__FILE__), (uint16_t)__LINE__, extra};           \
                assert_and_hang(__LINE__, debug_file_hash(__FILE__));                \
            } else {                                                                 \
                assert_and_hang(__LINE__, debug_file_hash(__FILE__), extra);         \
            }                                                                        \
        }                                                                            \
    } while (0)
#define _ASSERT_PLAIN(condition)                                  \
    do {                                                          \
        if (not(condition))                                       \
            assert_and_hang(__LINE__, debug_file_hash(__FILE__)); \
    } while (0)
#define _ASSERT_PICK(_c, _extra, _name, ...) _name

#define ASSERT(condition, ...) \
    _ASSERT_PICK(condition, ##__VA_ARGS__, _ASSERT_EXTRA, _ASSERT_PLAIN)(condition, ##__VA_ARGS__)

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
