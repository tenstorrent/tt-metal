// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "accessor/tensor_accessor.h"

void kernel_main() {
    auto args = TensorAccessorArgs<0, 0>();
    constexpr uint32_t base_idx_cta = args.compile_time_args_skip();
    uint32_t base_idx_crta = args.runtime_args_skip();

    constexpr uint32_t cb_id = get_compile_time_arg_val(base_idx_cta);
    constexpr uint32_t page_size = get_compile_time_arg_val(base_idx_cta + 1);

    const uint32_t bank_base_address = get_common_arg_val<uint32_t>(base_idx_crta);
    const uint32_t num_dev_pages = get_common_arg_val<uint32_t>(base_idx_crta + 1);

    auto tensor_accessor = TensorAccessor(args, bank_base_address, page_size);
    constexpr uint32_t one_tile = 1;
    for (uint32_t i = 0; i < num_dev_pages; ++i) {
        cb_wait_front(cb_id, one_tile);
        uint32_t l1_read_addr = get_read_ptr(cb_id);
        auto noc_addr = tensor_accessor.get_noc_addr(i);
        noc_async_write(l1_read_addr, noc_addr, page_size);
        noc_async_write_barrier();
        cb_pop_front(cb_id, one_tile);
    }
}
