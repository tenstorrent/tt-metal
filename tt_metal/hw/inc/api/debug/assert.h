// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/debug/watcher_common.h"
#include "internal/hw_thread.h"
#if defined(ARCH_QUASAR)
#include "internal/tt-2xx/quasar/error_handling.h"
#endif

// ASSERT must be defined before including risc_common.h. That header includes this file mid-way
// (so assert_and_hang can see flush_l2_cache_*), then uses ASSERT in ISR handlers. When this file
// is the include entry point (e.g. ckernel.h -> llk_assert.h -> here), the mid-file re-include is
// skipped by #pragma once, so ASSERT would otherwise be used before it is defined.
#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ASSERT) && !defined(FORCE_WATCHER_OFF)

// Definition below, after risc_common.h provides flush_l2_cache_range and related symbols.
inline void assert_and_hang(uint32_t line_num, debug_assert_type_t assert_type = DebugAssertTripped);

#define ASSERT(condition, ...) (void(not(condition) ? assert_and_hang(__LINE__, ##__VA_ARGS__), 0 : 0))

#define ASSERT_ENABLED 1
#define WATCHER_ASSERT_ENABLED 1
#define LIGHTWEIGHT_ASSERT_ENABLED 0

#elif defined(LIGHTWEIGHT_KERNEL_ASSERTS)

// Trap wrapped as a function to avoid inline assembly at ASSERT macro's use site.
FORCE_INLINE void lightweight_assert_trap() { asm("ebreak"); }

#define ASSERT(condition, ...) (void(not(condition) ? lightweight_assert_trap(), 0 : 0))

#define ASSERT_ENABLED 1
#define LIGHTWEIGHT_ASSERT_ENABLED 1
#define WATCHER_ASSERT_ENABLED 0

#else

// Avoid unused variable warnings here.
#define ASSERT(condition, ...) (void(sizeof(not(condition))))

#define ASSERT_ENABLED 0
#define LIGHTWEIGHT_ASSERT_ENABLED 0
#define WATCHER_ASSERT_ENABLED 0

#endif

// Included after ASSERT so risc_common.h's ISR handlers can use the macro even when this header
// is the include entry point (circular include with risc_common.h).
#include "risc_common.h"

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ASSERT) && !defined(FORCE_WATCHER_OFF)

//  - for Quasar, multiple DMs and TRISCs share assert_status; only the first to assert records its
//    metadata via a dedicated claim field atomically claimed (amoswap on the cached L1 alias).
//    Writes are flushed to make them visible to host.
inline void assert_and_hang(uint32_t line_num, debug_assert_type_t assert_type) {
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
#ifndef COMPILE_FOR_TRISC
            uint64_t mcause;
            uint64_t mtval;
            uint64_t mepc;
            asm volatile("csrr %0, mepc" : "=r"(mepc));
            asm volatile("csrr %0, mcause" : "=r"(mcause));
            asm volatile("csrr %0, mtval" : "=r"(mtval));
            v->line_num = mepc;  // mepc is the instruction address that caused the fault
            v->hw_fault_info = mtval << 32 | (mcause & 0xffffffff);  // mtval is the faulting address or instruction
#elif defined(ARCH_QUASAR) && defined(COMPILE_FOR_TRISC)
            // Layout constants and block IDs both come from error_handling.h, shared with the
            // host decoder so the two can't drift.
            //
            // Grab the register once. It's volatile, so a read per field is extra MMIO and can
            // end up mixing fields from different samples if another error shows up.
            uint32_t error_reg = RISC_PIC_BRISC_EX_REG_BASE(internal_::get_trisc_id())[HW_ERROR_INTERRUPT_INDEX];
            uint32_t neo_id = (error_reg >> kQuasarErrNeoShift) & kQuasarErrNeoMask;
            uint32_t block_id = (error_reg >> kQuasarErrBlockShift) & kQuasarErrBlockMask;
            v->hw_fault_info =
                (static_cast<uint64_t>(RISCV_DEBUG_REGS->ERR_DATA) << 32) | static_cast<uint64_t>(error_reg);
            // Only the per-TRISC blocks name a thread. The illegal-instruction ones count
            // backwards, so subtract. Everything else is Neo-level, put those on TRISC 0.
            uint32_t trisc_id = 0;
            if (block_id <= static_cast<uint32_t>(TriscErrors::ERROR_TRISC3)) {
                trisc_id = block_id;
            } else if (
                block_id >= static_cast<uint32_t>(TriscErrors::ILLEGAL_INSTRUCTION_TRISC3) &&
                block_id <= static_cast<uint32_t>(TriscErrors::ILLEGAL_INSTRUCTION_TRISC0)) {
                trisc_id = static_cast<uint32_t>(TriscErrors::ILLEGAL_INSTRUCTION_TRISC0) - block_id;
            }
            v->which = neo_id * NUM_TRISC_CORES + trisc_id + NUM_DM_CORES;
#endif
        }
        v->tripped = assert_type;
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
        // Flush the cache range covering debug_assert_msg_t to L1 so host sees all fields via NOC; may span multiple cache lines
        flush_l2_cache_range(reinterpret_cast<uintptr_t>(v), sizeof(debug_assert_msg_t));
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

#endif  // WATCHER_ENABLED
