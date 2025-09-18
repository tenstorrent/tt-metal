// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
This kernel copies local shards from one to another tensor. Output is required to have exactly the same spec.
*/

#include <cstdint>
#include "dataflow_api.h"
#include "accessor/tensor_accessor.h"

void kernel_main() {
    uint32_t page_size = get_compile_time_arg_val(0);
    uint32_t input_base_address = get_arg_val<uint32_t>(0);
    uint32_t output_base_address = get_arg_val<uint32_t>(1);
    uint32_t first_shard_id = get_arg_val<uint32_t>(2);
    uint32_t num_cores = get_arg_val<uint32_t>(3);
    uint32_t num_shards = get_arg_val<uint32_t>(4);

    auto args_src = TensorAccessorArgs<1, 0>();
    auto args_dst =
        TensorAccessorArgs<args_src.next_compile_time_args_offset(), args_src.next_common_runtime_args_offset()>();

    auto tensor_accessor_src = TensorAccessor(args_src, input_base_address, page_size);
    auto tensor_accessor_dst = TensorAccessor(args_dst, output_base_address, page_size);

    for (uint32_t i = 0; i < num_shards; ++i) {
        uint32_t shard_id = first_shard_id + i * num_cores;

#ifdef BIG_STEP
        for (uint32_t start_offset = 0; start_offset < BIG_STEP; ++start_offset) {
            auto shard_pages = tensor_accessor_src.shard_pages(shard_id, start_offset);
            auto it = shard_pages.begin();

            while (it != shard_pages.end()) {
                noc_async_write_page(it->page_id(), tensor_accessor_dst, it->noc_addr());
                noc_async_writes_flushed();
                it += BIG_STEP;
            }
        }

#else
        auto shard_pages = tensor_accessor_src.shard_pages(shard_id);
        for (const auto& page : shard_pages) {
            noc_async_write_page(page.page_id(), tensor_accessor_dst, page.noc_addr());
            noc_async_writes_flushed();
        }
#endif
    }
    noc_async_write_barrier();
}
