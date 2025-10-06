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

    experimental::Noc noc(noc_index);

    for (uint32_t i = 0; i < num_shards; ++i) {
        uint32_t shard_id = first_shard_id + i * num_cores;

        // For the purpose of tesing, every second shard is read, and every other is written.
        if (i % 2 == 0) {
            noc.async_read_shard(
                tensor_accessor_src,
                tensor_accessor_dst,
                {shard_id},
                // Use experimental::Noc::dst_args_t here will use get_noc_addr() instead
                // of get_shard_noc_addr() for dst.
                // The syntax seems a bit cumbersome.  Thoughts?
                // (Also, must specify type for this arg to avoid confusion)
                experimental::Noc::shard_args_t<decltype(tensor_accessor_dst)>{shard_id});
            noc.async_read_barrier();
        } else {
            noc.async_write_shard(
                tensor_accessor_src,
                tensor_accessor_dst,
                experimental::Noc::shard_args_t<decltype(tensor_accessor_src)>{shard_id},
                {shard_id});
            noc.async_write_barrier();
        }
    }
}
