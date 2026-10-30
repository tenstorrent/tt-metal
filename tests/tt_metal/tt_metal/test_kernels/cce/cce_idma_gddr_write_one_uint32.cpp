// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// cce_gddr_write_one_uint32.cpp through iDMA: the hart stores its compile-time value to its own TL1 staging word, then
// copies that word to each slot's GDDR word. Every hart of the CCE runs it at once, each on its own trid. On a timeout
// the kernel returns and the host's data check fails.

#include "api/compile_time_args.h"
#include "cce_idma_blocking_copy.h"
#include "experimental/cce_gddr.h"

void kernel_main() {
    constexpr uint32_t buffer_address = get_compile_time_arg_val(0);
    constexpr uint32_t mimir_index = get_compile_time_arg_val(1);
    constexpr uint32_t value = get_compile_time_arg_val(2);
    constexpr uint32_t staging_l1_address = get_compile_time_arg_val(3);
    constexpr uint32_t num_slots = get_compile_time_arg_val(4);
    constexpr uint32_t slot_stride = get_compile_time_arg_val(5);
    constexpr uint32_t slot_base = get_compile_time_arg_val(6);
    constexpr uint32_t cce_index = get_compile_time_arg_val(7);
    constexpr uint32_t backend_vc = get_compile_time_arg_val(8);
    constexpr uint32_t trid = get_compile_time_arg_val(9);

    *cce_test::own_tl1_words(staging_l1_address) = value;
    // The iDMA reads the staging word from TL1, so the store has to land first.
    asm volatile("fence" ::: "memory");

    const uint64_t staging_address = experimental::cce_sram_address(cce_index, staging_l1_address);
    const cce_test::WaitDeadline deadline(cce_test::kIdmaWaitBudgetCycles);
    for (uint32_t slot = 0; slot < num_slots; slot++) {
        const uint64_t offset = static_cast<uint64_t>(slot_base + slot) * slot_stride;
        if (!cce_test::idma_copy_and_wait(
                staging_address,
                experimental::cce_gddr_address(mimir_index, buffer_address + offset),
                sizeof(uint32_t),
                backend_vc,
                trid,
                deadline)) {
            return;
        }
    }
}
