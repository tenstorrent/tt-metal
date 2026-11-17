// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/local_tensor_accessor.h"

void kernel_main() {
    // This test uses legacy RTA indices to get the bank base addresses.
    // In Metal 2.0, these raw addresses are passed implicitly.
    uint32_t input_base_address = get_arg_val<uint32_t>(0);
    uint32_t output_base_address = get_arg_val<uint32_t>(1);
    // Args (2) first_shard_id and (3) num_cores are unused — the bank-base copy is bank-relative.
    uint32_t n_shards = get_arg_val<uint32_t>(4);

    // Only the source's distribution spec is needed (for the shard size); the destination has the same layout.
    auto args_src = TensorAccessorArgs<1, 0>();
    auto tensor_accessor_src = TensorAccessor(args_src, input_base_address);

    // For L1-sharded tensors the bank base address is the local L1 address of this core's shard region,
    // so each tensor's local region is a LocalTensorAccessor over it.
    // In Metal 2.0, this would just be LocalTensorAccessor<T>(tensor::my_accessor_name).
    const LocalTensorAccessor<uint32_t> src_local(input_base_address);
    const LocalTensorAccessor<uint32_t> dst_local(output_base_address);
    const uint32_t shard_size_bytes =
        tensor_accessor_src.get_aligned_page_size() * tensor_accessor_src.dspec().shard_volume();
    const uint32_t total_bytes = shard_size_bytes * n_shards;

    // One NoC read (loopback from this core's own L1) covers n_shards consecutive shard-sized regions.
    Noc noc(noc_index);
    noc.async_read(src_local, dst_local, total_bytes, {.offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
}
