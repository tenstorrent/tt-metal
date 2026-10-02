// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "api/debug/device_print.h"

/*
 * Thread 0 takes the print lock through the production code, raises a flag, lets thread 1 spin on
 * the lock for a while and then releases it through the production code. Thread 1 spins on the
 * lock the way acquire_lock() does and reports whether it got the lock and after how many attempts.
 * Quasar DM only; built with DEBUG_PRINT_ENABLED so the lock code is compiled in.
 */

void kernel_main() {
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM) && defined(DEBUG_PRINT_ENABLED)
    constexpr uint32_t done_marker = 0x52454C53u;
    constexpr uint32_t max_attempts = 1u << 20;

    auto uncached = [](uint32_t addr) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(addr) + MEM_L1_UNCACHED_BASE);
    };
    volatile tt_l1_ptr uint32_t* report = uncached(get_arg(args::report_addr));
    volatile tt_l1_ptr uint32_t* flag = uncached(get_arg(args::flag_addr));
    const bool prints_off = get_arg(args::prints_off) != 0;
    auto& lock = device_print_detail::locking::get_lock_atomic();
    volatile tt_l1_ptr DevicePrintBufferType* header = get_device_print_buffer();

    if (get_my_thread_id() == 0) {
        const uint32_t saved_wpos = header->aux.wpos;
        if (prints_off) {
            header->aux.wpos = DEBUG_PRINT_SERVER_DISABLED_MAGIC;
        }
        lock.exchange(0u);
        device_print_detail::locking::acquire_lock();
        __asm__ __volatile__("fence" ::: "memory");
        *flag = 1u;  // thread 1 starts spinning on the lock
        for (volatile uint32_t i = 0; i < 4096u; i++) {
        }
        device_print_detail::locking::release_lock();
        if (prints_off) {
            header->aux.wpos = saved_wpos;
        }
        report[2] = internal_::get_hw_thread_idx();
        report[0] = done_marker;
    } else if (get_my_thread_id() == 1) {
        uint32_t spins = 1u << 22;
        while (*flag != 1u && spins != 0) {
            spins--;
        }
        report[3] = internal_::get_hw_thread_idx();
        if (spins == 0) {
            report[1] = 2u;  // timed out waiting for thread 0 to take the lock
            return;
        }
        uint32_t attempts = 1;
        while (lock.exchange(1u) != 0u && attempts < max_attempts) {
            attempts++;
        }
        report[4] = attempts;
        if (attempts == max_attempts) {
            report[1] = 3u;  // never saw the lock free
            return;
        }
        lock.exchange(0u);
        report[1] = 1u;
    }
#endif
}
