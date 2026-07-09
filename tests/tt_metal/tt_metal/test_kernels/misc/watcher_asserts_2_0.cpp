// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 host-API version of watcher_asserts.cpp.
// Compiled only for TENSIX cores (BRISC / NCRISC / TRISC / DM). Ethernet and DRAM
// callers continue to use watcher_asserts.cpp via the legacy host API.

#include <cstdint>
#include "api/debug/assert.h"
#include "api/debug/ring_buffer.h"
#include "api/kernel_thread_globals.h"
#include "internal/firmware_common.h"
#include "experimental/kernel_args.h"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#endif

void kernel_main() {
    uint32_t a = get_arg(args::a);
    uint32_t b = get_arg(args::b);
    uint32_t assert_type = get_arg(args::assert_type);

#if defined(COMPILE_FOR_DM)
    // On Quasar all user DMs are launched; run only the one whose kernel-local
    // thread id matches the host-supplied target.
    constexpr uint32_t target_thread_id = get_arg(args::target_thread_id);
    if (get_my_thread_id() != target_thread_id) {
        return;
    }
#endif

#if (defined(UCK_CHLKC_UNPACK) and defined(TRISC0)) or (defined(UCK_CHLKC_MATH) and defined(TRISC1)) or       \
    (defined(UCK_CHLKC_PACK) and defined(TRISC2)) or (defined(UCK_CHLKC_ISOLATE_SFPU) and defined(TRISC3)) or \
    (defined(COMPILE_FOR_BRISC) or defined(COMPILE_FOR_NCRISC) or defined(COMPILE_FOR_DM))
    WATCHER_RING_BUFFER_PUSH(a);
    WATCHER_RING_BUFFER_PUSH(b);
#if defined(COMPILE_FOR_BRISC) or defined(COMPILE_FOR_NCRISC) or defined(COMPILE_FOR_DM)
    if (a == b) {
        // Signal completion to dispatcher before the assert hangs the kernel so the
        // dispatcher (and Device::close) can still make progress.
        volatile tt_l1_ptr go_msg_t* go_message_in = GET_MAILBOX_ADDRESS_DEV(go_messages[0]);
        go_message_in->signal = RUN_MSG_DONE;
        uint64_t dispatch_addr = calculate_dispatch_addr(go_message_in);
        notify_dispatch_core_done(dispatch_addr, noc_index);
    }
#else
#if defined(COMPILE_FOR_TRISC)
#if defined(ARCH_QUASAR)
    uint32_t hw_idx = internal_::get_hw_thread_idx();
    volatile tt_l1_ptr uint8_t* const trisc_run =
        &((tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE))->subordinate_sync.map[hw_idx];
#else
    volatile tt_l1_ptr uint8_t* const trisc_run =
        &((tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE))
             ->subordinate_sync.map[COMPILE_FOR_TRISC + 1];  // first entry is for NCRISC
#endif
    *trisc_run = RUN_SYNC_MSG_DONE;
#endif
#endif
    if (assert_type == DebugAssertHwFault && a == b) {
        uint32_t hw_assert_cause = get_arg(args::hw_assert_cause);
#ifndef COMPILE_FOR_TRISC
        volatile int32_t* p = (int32_t*)0xffffffffff000000;
#else
#if defined(ARCH_QUASAR)
        // On Quasar the kernel runs on all TRISCs; bail out on the ones whose
        // local trisc index doesn't match the host-supplied target.
        constexpr uint32_t compute_id = get_arg(args::trisc_id);
        if ((hw_idx - NUM_DM_CORES) != compute_id) {  // test always runs on neo0 cluster
            return;
        }
#endif
        volatile int32_t* p = (int32_t*)0xff000000;
#endif
        uint32_t tmp;
        switch (hw_assert_cause) {
            case 2: asm volatile(".word 0x00000000"); break;            // illegal instruction
            case 4: asm volatile("lw %0, 0x2(x0)" : "=r"(tmp)); break;  // load not aligned
            case 5: tmp = *p; break;                                    // load access fault
            case 6: asm volatile("sw %0, 0x2(x0)" : "=r"(tmp)); break;  // store not aligned
            case 7: *p = 0; break;                                      // store access fault
#if defined(COMPILE_FOR_TRISC) and defined(ARCH_QUASAR)
            case 8: INSTRUCTION_WORD(TT_OP(0xbc, 0)); break;  // illegal instruction
            case 9:
                RISCV_DEBUG_REGS->CHICKEN_BITS |= T6_DEBUG_REGS__CHICKEN_BITS__ALLOW_UNSAFE_SEMPOST_SEMGET_bm;
                TTI_SEMGET(0, 2);
                TTI_SEMINIT(3, 0, 0, 2);
                break;  // semaphore
            case 10: {
                uint32_t* stack_limit = (uint32_t*)RISCV_DEBUG_REGS->DBG_TRISC_STACK_LIMIT;
                volatile uint32_t stack_value = *(stack_limit - 1);
                break;  // stack overflow
            }
#endif
            default: ASSERT(0, DebugAssertHwFault);
        }
    } else {
        ASSERT(a != b, static_cast<debug_assert_type_t>(assert_type));
    }
#endif
}
