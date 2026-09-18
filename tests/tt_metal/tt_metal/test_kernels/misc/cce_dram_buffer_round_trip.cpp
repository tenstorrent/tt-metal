// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dev_mem_map.h"
#include "experimental/cce_gddr.h"
#include "risc_common.h"

void kernel_main() {
    const uint32_t src_buffer_address = get_arg_val<uint32_t>(0);
    const uint32_t dst_buffer_address = get_arg_val<uint32_t>(1);
    const uint32_t dram_partition = get_arg_val<uint32_t>(2);
    const uint32_t staging_l1_address = get_arg_val<uint32_t>(3) + MEM_L1_UNCACHED_BASE;
    const uint32_t num_words = get_arg_val<uint32_t>(4);

    experimental::cce_gddr_read(dram_partition, src_buffer_address, staging_l1_address, num_words);
    experimental::cce_gddr_write(dram_partition, dst_buffer_address, staging_l1_address, num_words);
}
