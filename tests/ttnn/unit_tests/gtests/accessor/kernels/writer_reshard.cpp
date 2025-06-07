// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "accessor/sharded_accessor.h"

void kernel_main() {
    const uint32_t bank_base_address = get_common_arg_val<uint32_t>(0);

    // The compile-time args are set up like this to highlight how you can use compile_time_args_skip
    // Recommended usage is to place the sequential compile-time args for distribution spec at the end
    constexpr uint32_t base_idx_cta = 0;
    constexpr uint32_t base_idx_crta = 1;

    using output_dspec = nd_sharding::distribution_spec_t<base_idx_cta, base_idx_crta>;
    constexpr uint32_t new_base_idx_cta = base_idx_cta + nd_sharding::compile_time_args_skip<output_dspec>();
    constexpr uint32_t new_base_idx_crta = base_idx_crta + nd_sharding::runtime_args_skip<output_dspec>();

    constexpr uint32_t cb_id = get_compile_time_arg_val(new_base_idx_cta);
    // TODO: Expose generic interface to get page size for cb operand
    // - get_tile_size(cb_id) only works for tile layout
    // In writer page size is a compile-time argument
    constexpr uint32_t page_size = get_compile_time_arg_val(new_base_idx_cta + 1);

    auto sharded_accessor = nd_sharding::ShardedAccessor<output_dspec, page_size>(bank_base_address);
    constexpr uint32_t rank = sharded_accessor.get_dspec().get_rank();
    constexpr uint32_t num_banks = sharded_accessor.get_dspec().get_num_banks();

    constexpr uint32_t one_tile = 1;
    for (size_t i = 0; i < sharded_accessor.get_dspec().get_tensor_volume(); ++i) {
        cb_wait_front(cb_id, one_tile);
        uint32_t l1_read_addr = get_read_ptr(cb_id);
        sharded_accessor.noc_async_write_page(i, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id, one_tile);
    }
}
