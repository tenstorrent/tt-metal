// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
Strided shard pages copy kernel — whole-shard granularity.

Each core (simulating one DM thread) copies the shards assigned to it:
  shards at indices: first_shard_id, first_shard_id + num_cores, ...

Runtime args (slots 0–4):
  0: input_base_address
  1: output_base_address
  2: first_shard_id   — first shard index owned by this core (== core rank / tid)
  3: num_cores        — stride between consecutive shards owned by the same core
  4: n_shards         — number of shards assigned to THIS core

total_shards sentinel passed to StridedShardPages is derived as:
  first_shard_id + n_shards * num_cores
This is the first shard index that would exceed this core's assignment, making the
range equivalent to { first_shard_id, first_shard_id+num_cores, ... } for n_shards steps.

Uses StridedShardPages proxy to iterate over the assigned shard subset.
*/

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    uint32_t aligned_page_size = get_compile_time_arg_val(0);
    auto args_src = TensorAccessorArgs<1, 0>();
    auto args_dst =
        TensorAccessorArgs<args_src.next_compile_time_args_offset(), args_src.next_common_runtime_args_offset()>();

    uint32_t input_base_address  = get_arg_val<uint32_t>(0);
    uint32_t output_base_address = get_arg_val<uint32_t>(1);
    uint32_t first_shard_id      = get_arg_val<uint32_t>(2);
    uint32_t num_cores           = get_arg_val<uint32_t>(3);
    uint32_t n_shards            = get_arg_val<uint32_t>(4);  // per-core count, not global total

    // Derive the sentinel: one stride past the last shard this core owns.
    uint32_t total_shards = first_shard_id + n_shards * num_cores;

    auto tensor_accessor_src = TensorAccessor(args_src, input_base_address, aligned_page_size);
    auto tensor_accessor_dst = TensorAccessor(args_dst, output_base_address, aligned_page_size);

    // Iterate over shards assigned to this core: first_shard_id, first_shard_id+num_cores, ...
    tensor_accessor::StridedShardPages strided_shards(
        tensor_accessor_src, first_shard_id, total_shards, num_cores);

    for (const auto& shard_range : strided_shards) {
        for (const auto& page : shard_range) {
            noc_async_write_page(page.page_id(), tensor_accessor_dst, page.noc_addr());
            noc_async_writes_flushed();
        }
    }
    noc_async_write_barrier();
}
