// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "accessor/sharded_accessor.h"

void kernel_main() {
    const uint32_t bank_base_address = get_arg_val<uint32_t>(0);

    // The compile-time args are set up like this to highlight how you can use compile_time_args_skip
    // Recommended usage is to place the sequential compile-time args for distribution spec at the end
    constexpr uint32_t rank = get_compile_time_arg_val(0);
    constexpr uint32_t num_banks = get_compile_time_arg_val(1);
    constexpr uint32_t base_idx = 2;

    using input_dspec = distribution_spec_t<base_idx, rank, num_banks>;
    constexpr uint32_t new_base_idx = base_idx + compile_time_args_skip<input_dspec>;

    constexpr uint32_t cb_id = get_compile_time_arg_val(new_base_idx);
    // TODO: Expose generic interface to get page size for cb operand
    // - get_tile_size(cb_id) only works for tile layout
    constexpr uint32_t page_size = get_compile_time_arg_val(new_base_idx + 1);

    auto sharded_accessor = ShardedAccessor<input_dspec, page_size>{.bank_base_address = bank_base_address};

    constexpr uint32_t one_tile = 1;
    for (size_t i = 0; i < input_dspec::tensor_volume; ++i) {
        cb_reserve_back(cb_id, one_tile);
        uint32_t l1_write_addr = get_write_ptr(cb_id);
        sharded_accessor.noc_async_read_page(i, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id, one_tile);
    }
}
