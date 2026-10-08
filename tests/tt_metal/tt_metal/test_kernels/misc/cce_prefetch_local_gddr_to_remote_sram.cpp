// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dev_mem_map.h"
#include "experimental/cce_gddr.h"
#include "risc_common.h"

void kernel_main() {
    const uint32_t src_gddr_address = get_arg_val<uint32_t>(0);
    const uint32_t local_mimir = get_arg_val<uint32_t>(1);
    const uint32_t staging_l1_address = get_arg_val<uint32_t>(2);
    const uint32_t num_words = get_arg_val<uint32_t>(3);
    const uint32_t dst0 = get_arg_val<uint32_t>(4);
    const uint32_t dst1 = get_arg_val<uint32_t>(5);
    const uint32_t sram_offset = get_arg_val<uint32_t>(6);
    const uint32_t gddr_base_slots = get_arg_val<uint32_t>(7);
    const uint32_t gddr_slot_stride = get_arg_val<uint32_t>(8);
    const uint32_t staging_uncached_address = staging_l1_address + MEM_L1_UNCACHED_BASE;
    const uint64_t gddr_address = src_gddr_address + static_cast<uint64_t>(gddr_base_slots) * gddr_slot_stride;

    experimental::cce_gddr_read(local_mimir, local_mimir, gddr_address, staging_uncached_address, num_words);
    experimental::cce_sram_write(dst0, sram_offset, staging_uncached_address, num_words);
    experimental::cce_sram_write(dst1, sram_offset, staging_uncached_address, num_words);
}
