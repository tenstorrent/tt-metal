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

    auto args_src = make_tensor_accessor_args<1, 0>();
    constexpr uint32_t base_idx_cta_src = 1 + args_src.compile_time_args_skip();
    constexpr uint32_t base_idx_crta_src = args_src.runtime_args_skip();

    auto args_dst = make_tensor_accessor_args<base_idx_cta_src, base_idx_crta_src>();

    auto tensor_accessor_src = make_tensor_accessor_from_args(args_src, input_base_address, page_size);
    auto tensor_accessor_dst = make_tensor_accessor_from_args(args_dst, output_base_address, page_size);

    for (uint32_t i = 0; i < num_shards; ++i) {
        uint32_t shard_id = first_shard_id + i * num_cores;
        auto noc_addr_src = tensor_accessor_src.get_shard_noc_addr(shard_id);
        auto noc_addr_dst = tensor_accessor_dst.get_shard_noc_addr(shard_id);

        ASSERT(tensor_accessor_src.is_local_shard(shard_id));
        ASSERT(tensor_accessor_dst.is_local_shard(shard_id));
        ASSERT(tensor_accessor_src.is_local_addr(noc_addr_src));
        ASSERT(tensor_accessor_dst.is_local_addr(noc_addr_dst));

        // For the purpose of tesing, every second shard is read, and every other is written.
        if (i % 2 == 0) {
            noc_async_read_shard(shard_id, tensor_accessor_src, noc_addr_dst);
        } else {
            noc_async_write_shard(shard_id, tensor_accessor_dst, noc_addr_src);
        }
    }
    noc_async_read_barrier();
}
