// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_time_args.h"
#include "dev_mem_map.h"
#include "experimental/cce_gddr.h"

void kernel_main() {
    constexpr uint32_t src_buffer_address = get_compile_time_arg_val(0);
    constexpr uint32_t dst_buffer_address = get_compile_time_arg_val(1);
    constexpr uint32_t mimir_index = get_compile_time_arg_val(2);
    constexpr uint32_t staging_l1_address = get_compile_time_arg_val(3);
    constexpr uint32_t num_words = get_compile_time_arg_val(4);
    constexpr uint32_t num_slots = get_compile_time_arg_val(5);
    constexpr uint32_t slot_stride = get_compile_time_arg_val(6);
    constexpr uint32_t slot_base = get_compile_time_arg_val(7);
    constexpr uint32_t staging_uncached_address = staging_l1_address + MEM_L1_UNCACHED_BASE;

    for (uint32_t slot = 0; slot < num_slots; slot++) {
        const uint64_t offset = static_cast<uint64_t>(slot_base + slot) * slot_stride;
        experimental::cce_gddr_read(mimir_index, src_buffer_address + offset, staging_uncached_address, num_words);
        experimental::cce_gddr_write(mimir_index, dst_buffer_address + offset, staging_uncached_address, num_words);
    }
}
