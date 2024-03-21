// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "tt_eager/tt_dnn/op_library/all_gather/kernels/dataflow/worker_ring_gather_utils.hpp"

void kernel_main() {
    constexpr ShardType shard_type = static_cast<ShardType>(get_compile_time_arg_val(0));

    ShardAddrGen<shard_type> output_tensor_shard_writer;
    uint32_t arg_index = 0;
    uint32_t const remote_sender_worker_x = get_arg_val<uint32_t>(arg_index++);
    uint32_t const remote_sender_worker_y = get_arg_val<uint32_t>(arg_index++);
    uint32_t const remote_sender_reader_semaphore_addres = get_arg_val<uint32_t>(arg_index++);
    uint32_t const max_shards_per_eth_buffer = get_arg_val<uint32_t>(arg_index++);
    uint32_t const num_transfers = get_arg_val<uint32_t>(arg_index++);
    ShardAddrGen<shard_type>::build_with_placement_new(&output_tensor_shard_writer, arg_index);
    arg_index += output_tensor_shard_writer.get_num_args_consumed();

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    // Each worker receiver writer matches with a specific worker sender reader
    // Used to signal that data has been committed to memory and can be read
    const uint64_t worker_send_reader_semaphore_noc_addr =
        get_noc_addr(remote_sender_worker_x, remote_sender_worker_y, remote_sender_reader_semaphore_addres);

    uint32_t total_num_shards = output_tensor_shard_writer.get_num_dest_cores() *
                             (output_tensor_shard_writer.get_chunks_per_core_before_advance() - 1);

    for (uint32_t d = 0; d < num_transfers; d++) {
        uint32_t num_shards_to_send = std::min(max_shards_per_eth_buffer, total_num_shards);

        write_chunk_sharded(cb_id_in0, output_tensor_shard_writer, num_shards_to_send);  // 1 shard = 1 page

        noc_semaphore_inc(worker_send_reader_semaphore_noc_addr, num_shards_to_send);
        total_num_shards -= num_shards_to_send;

        ASSERT(total_num_shards > 0 || d == num_transfers - 1); // If we are out of shards, make sure we are on the last transfer
    }
}
