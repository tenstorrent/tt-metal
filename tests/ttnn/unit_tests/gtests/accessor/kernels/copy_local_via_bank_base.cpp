// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    // This test uses legacy RTA indices to get the bank base addresses.
    // In Metal 2.0, these raw addresses are passed implicitly.
    uint32_t input_base_address = get_arg_val<uint32_t>(0);
    uint32_t output_base_address = get_arg_val<uint32_t>(1);
    // Args (2) first_shard_id and (3) num_cores are unused — the bank-base copy is bank-relative.
    uint32_t n_shards = get_arg_val<uint32_t>(4);

    auto args_src = TensorAccessorArgs<1, 0>();
    auto args_dst =
        TensorAccessorArgs<args_src.next_compile_time_args_offset(), args_src.next_common_runtime_args_offset()>();

    // In Metal 2.0, this would just be TensorAccessor(tensor::my_accessor_name)
    // The bank base address would not be available in scope.
    auto tensor_accessor_src = TensorAccessor(args_src, input_base_address);
    auto tensor_accessor_dst = TensorAccessor(args_dst, output_base_address);

    // If you needed the pointer, you could pull it out of the TensorAccessor:
    const uint32_t src_l1 = tensor_accessor_src.get_bank_base_address();
    const uint32_t dst_l1 = tensor_accessor_dst.get_bank_base_address();
    const uint32_t shard_size_bytes =
        tensor_accessor_src.get_aligned_page_size() * tensor_accessor_src.dspec().shard_volume();
    const uint32_t total_bytes = shard_size_bytes * n_shards;

    // For L1-sharded tensors get_bank_base_address() returns the local L1 address of this core's
    // shard region. One NoC read covers n_shards consecutive shard-sized regions.
    noc_async_read(get_noc_addr(src_l1), dst_l1, total_bytes);
    noc_async_read_barrier();
}
