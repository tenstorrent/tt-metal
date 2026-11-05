// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "hostdevcommon/dprint_common.h"
#include "internal/debug/dprint_buffer.h"

/*
 * Dirty the print lock's cache line with the lock AMO, write new header values through the
 * uncached alias, write the lock's line back, and report whether the header kept the new values.
 * Quasar DM only.
 */

void kernel_main() {
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    constexpr uint32_t num_words = 16;  // wpos, rpos, risc_state (2 words), data[0..11]
    constexpr uint32_t done_marker = 0x4C4F434Bu;

    volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        static_cast<uintptr_t>(get_arg(args::report_addr)) + MEM_L1_UNCACHED_BASE);
    auto& lock = GET_MAILBOX_ADDRESS_DEV_CACHED(dprint_buf.buffer)->aux.lock;
    volatile tt_l1_ptr DevicePrintBufferType* header = get_device_print_buffer();

    volatile tt_l1_ptr uint32_t* word[num_words];
    word[0] = &header->aux.wpos;
    word[1] = &header->aux.rpos;
    word[2] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(&header->aux.risc_state[0]);
    word[3] = word[2] + 1;
    for (uint32_t i = 4; i < num_words; i++) {
        word[i] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(&header->data[0]) + (i - 4);
    }

    uint32_t saved[num_words];
    for (uint32_t i = 0; i < num_words; i++) {
        saved[i] = *word[i];
        *word[i] = 0;
    }
    __asm__ __volatile__("fence" ::: "memory");

    lock.exchange(1u);
    for (uint32_t i = 0; i < num_words; i++) {
        *word[i] = 0xA5A50000u + i;
    }
    __asm__ __volatile__("fence" ::: "memory");

    flush_l2_cache_line(reinterpret_cast<uintptr_t>(&lock));

    uint32_t changed = 0;
    for (uint32_t i = 0; i < num_words; i++) {
        const uint32_t v = *word[i];
        report[2 + i] = v;
        changed += (v != 0xA5A50000u + i) ? 1u : 0u;
    }

    lock.exchange(0u);
    flush_l2_cache_line(reinterpret_cast<uintptr_t>(&lock));
    for (uint32_t i = 0; i < num_words; i++) {
        *word[i] = saved[i];
    }
    __asm__ __volatile__("fence" ::: "memory");

    report[1] = changed;
    report[0] = done_marker;
#endif
}
