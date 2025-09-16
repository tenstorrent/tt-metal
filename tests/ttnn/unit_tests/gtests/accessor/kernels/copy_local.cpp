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

    DPRINT << "SRC DSPEC" << ENDL();
    {
        auto dspec = tensor_accessor_src.dspec();
        DPRINT << "Base address: " << input_base_address << ENDL();
        DPRINT << "Rank: " << dspec.rank() << ENDL();
        auto physical_num_banks = dspec.num_banks();
        DPRINT << "Number of banks: " << physical_num_banks << ENDL();
        for (uint32_t i = 0; i < dspec.rank(); ++i) {
            DPRINT << "Tensor shape[" << i << "]: " << dspec.tensor_shape()[i] << ENDL();
        }
        for (uint32_t i = 0; i < dspec.rank(); ++i) {
            DPRINT << "Shard shape[" << i << "]: " << dspec.shard_shape()[i] << ENDL();
        }
        for (uint32_t i = 0; i < physical_num_banks; ++i) {
            auto xy = dspec.packed_xy_coords()[i];
            auto x = (xy >> 8) & 0xFF;
            auto y = xy & 0xFF;
            DPRINT << "Bank coords[" << i << "]: " << x - 18 << ", " << y - 18 << ENDL();
        }
    }
    DPRINT << "DST DSPEC" << ENDL();
    {
        auto dspec_b = tensor_accessor_dst.dspec();
        DPRINT << "Base address: " << output_base_address << ENDL();
        DPRINT << "Rank: " << dspec_b.rank() << ENDL();
        auto physical_num_banks_b = dspec_b.num_banks();
        DPRINT << "Number of banks: " << physical_num_banks_b << ENDL();
        for (uint32_t i = 0; i < dspec_b.rank(); ++i) {
            DPRINT << "Tensor shape[" << i << "]: " << dspec_b.tensor_shape()[i] << ENDL();
        }
        for (uint32_t i = 0; i < dspec_b.rank(); ++i) {
            DPRINT << "Shard shape[" << i << "]: " << dspec_b.shard_shape()[i] << ENDL();
        }
        for (uint32_t i = 0; i < physical_num_banks_b; ++i) {
            auto xy = dspec_b.packed_xy_coords()[i];
            auto x = (xy >> 8) & 0xFF;
            auto y = xy & 0xFF;
            DPRINT << "Bank coords[" << i << "]: " << x - 18 << ", " << y - 18 << ENDL();
        }
    }
    DPRINT << "Num shards: " << (uint32_t)num_shards << ENDL();
    DPRINT << "Num cores: " << (uint32_t)num_cores << ENDL();
    DPRINT << "First shard id: " << (uint32_t)first_shard_id << ENDL();

    for (uint32_t i = 0; i < num_shards; ++i) {
        uint32_t shard_id = first_shard_id + i * num_cores;
        DPRINT << "Shard id: " << shard_id << ENDL();
        auto noc_addr_src = tensor_accessor_src.get_shard_noc_addr(shard_id);
        auto noc_addr_dst = tensor_accessor_dst.get_shard_noc_addr(shard_id);
        DPRINT << "Noc addr src: " << noc_addr_src << ENDL();
        DPRINT << "Noc addr dst: " << noc_addr_dst << ENDL();

        ASSERT(tensor_accessor_src.is_local_shard(shard_id));
        ASSERT(tensor_accessor_dst.is_local_shard(shard_id));
        ASSERT(tensor_accessor_src.is_local_addr(noc_addr_src));
        ASSERT(tensor_accessor_dst.is_local_addr(noc_addr_dst));

        // For the purpose of tesing, every second shard is read, and every other is written.
        noc_async_read_shard(shard_id, tensor_accessor_src, noc_addr_dst);
        // if (i % 2 == 0) {
        // } else {
        //     noc_async_write_shard(shard_id, tensor_accessor_dst, noc_addr_src);
        // }
    }
    noc_async_read_barrier();
}
