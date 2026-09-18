// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_time_args.h"
#include "dev_mem_map.h"
#include "experimental/cce_gddr.h"

void kernel_main() {
    constexpr uint32_t buffer_address = get_compile_time_arg_val(0);
    constexpr uint32_t dram_partition = get_compile_time_arg_val(1);
    constexpr uint32_t value = get_compile_time_arg_val(2);
    constexpr uint32_t staging_l1_address = get_compile_time_arg_val(3) + MEM_L1_UNCACHED_BASE;

    volatile tt_l1_ptr uint32_t* staging = reinterpret_cast<tt_l1_ptr uint32_t*>(staging_l1_address);
    staging[0] = value;
    experimental::cce_gddr_write(dram_partition, buffer_address, staging_l1_address, 1);
}
