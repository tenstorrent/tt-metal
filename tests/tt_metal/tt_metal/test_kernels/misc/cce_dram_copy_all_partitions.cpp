// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dev_mem_map.h"
#include "experimental/cce_gddr.h"
#include "risc_common.h"

void kernel_main() {
    const uint32_t src_buffer_address = get_arg_val<uint32_t>(0);
    const uint32_t dst_buffer_address = get_arg_val<uint32_t>(1);
    const uint32_t num_mimirs = get_arg_val<uint32_t>(2);
    const uint32_t staging_l1_address = get_arg_val<uint32_t>(3);
    const uint32_t num_words = get_arg_val<uint32_t>(4);
    const uint32_t staging_uncached_address = staging_l1_address + MEM_L1_UNCACHED_BASE;

    for (uint32_t mimir_index = 0; mimir_index < num_mimirs; mimir_index++) {
        experimental::cce_gddr_read(mimir_index, src_buffer_address, staging_uncached_address, num_words);
        experimental::cce_gddr_write(mimir_index, dst_buffer_address, staging_uncached_address, num_words);
    }
}
