// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_time_args.h"
#include "dev_mem_map.h"
#include "experimental/cce_gddr.h"

void kernel_main() {
    constexpr uint32_t buffer_address = get_compile_time_arg_val(0);
    constexpr uint32_t mimir_index = get_compile_time_arg_val(1);
    constexpr uint32_t value = get_compile_time_arg_val(2);
    constexpr uint32_t staging_l1_address = get_compile_time_arg_val(3);
    constexpr uint32_t num_slots = get_compile_time_arg_val(4);
    constexpr uint32_t slot_stride = get_compile_time_arg_val(5);
    constexpr uint32_t slot_base = get_compile_time_arg_val(6);

    volatile tt_l1_ptr uint32_t* staging =
        reinterpret_cast<tt_l1_ptr uint32_t*>(staging_l1_address + MEM_L1_UNCACHED_BASE);
    staging[0] = value;
    for (uint32_t slot = 0; slot < num_slots; slot++) {
        const uint64_t offset = static_cast<uint64_t>(slot_base + slot) * slot_stride;
        experimental::cce_gddr_write(
            mimir_index, buffer_address + offset, staging_l1_address + MEM_L1_UNCACHED_BASE, 1);
    }
}
