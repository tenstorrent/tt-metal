// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

#include "../rt_args_common.hpp"

// Helper template to get shard noc address - uses type-dependent expression
// to defer name lookup until instantiation (avoids compile error on interleaved)
template <typename Accessor>
FORCE_INLINE uint64_t get_shard_noc_addr_helper(const Accessor& reader, uint32_t shard_id) {
    return reader.get_shard_noc_addr(shard_id);
}

// Semaphore values for multicast synchronization
constexpr uint32_t MCAST_INVALID = 0;
constexpr uint32_t MCAST_VALID = 1;

/******************************************************************************
 *                   Kernel Main                                               *
 ******************************************************************************/
void kernel_main() {
    /*
    Simplified Flash MLA Decode reader kernel for Deepseek V3 B1.
    Assumptions:
    - PNHt = 1 (single tile of Q heads per core)
    - num_kv_heads = 1 (MLA has single KV head)
    - Bkv = 1 (single batch for KV cache)
    - Q is always sharded, KV cache is always ND-sharded in DRAM
    */
    constexpr uint32_t St = get_compile_time_arg_val(0);                  // full sequence length of kv cache in tiles
    constexpr uint32_t DHt = get_compile_time_arg_val(1);                 // head dim in tiles (K width)
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(2);          // tiles per K chunk
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(3);  // cores for seq len parallelism (8)
    constexpr uint32_t k_chunk_size = get_compile_time_arg_val(4);
    constexpr uint32_t q_chunk_size_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t num_mcast_dests = get_compile_time_arg_val(6);
    constexpr uint32_t mcast_semaphore_id = get_compile_time_arg_val(7);
    constexpr uint32_t k_page_size = get_compile_time_arg_val(8);
    constexpr uint32_t k_num_pages = get_compile_time_arg_val(9);
    constexpr uint32_t ncrisc_brisc_sync_semaphore_id = get_compile_time_arg_val(10);
    constexpr uint32_t receiver_ready_semaphore_id = get_compile_time_arg_val(11);
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(12);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(13);

    // TensorAccessorArgs for K (KV cache in DRAM) only - position is read directly from sharded L1
    constexpr auto k_args = TensorAccessorArgs<14>();

    uint32_t arg_idx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pos_addr = get_arg_val<uint32_t>(arg_idx++);  // Position is height-sharded in L1
    const bool is_output_core = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t cur_batch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const bool is_mcast_sender = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t mcast_start_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_start_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_end_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_end_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t vc = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_core_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_core_noc_y = get_arg_val<uint32_t>(arg_idx++);

    // Get cur_pos from height-sharded position tensor (directly from local L1)
    // Position tensor is replicated on every core - just read from the local shard address
    volatile tt_l1_ptr uint32_t* pos_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_addr);
    uint32_t cur_pos = pos_ptr[0];

    // Sequence length assignment (no sliding window for MLA)
    auto [k_num_chunks, k_chunk_start, k_chunk_end] =
        get_runtime_args(cur_pos, cur_batch, core_num_in_reduce, num_cores_per_head, k_chunk_size);
    (void)k_num_chunks;  // Unused in reader

    if (k_chunk_start == k_chunk_end) {
        return;  // early exit because no compute needs to be done
    }

    // PNHt = 1, so q_chunk_tiles = DHt
    constexpr uint32_t q_chunk_tiles = DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;

    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);

    // Read Q from sharded memory (Q is always sharded and tilized)
    {
        DeviceZoneScopedN("reader-q-read");
        uint64_t q_read_addr;
        if (is_output_core) {
            q_read_addr = get_noc_addr(q_addr);
        } else {
            q_read_addr = get_noc_addr(output_core_noc_x, output_core_noc_y, q_addr);
        }
        cb_reserve_back(cb_q_in, q_chunk_tiles);
        uint32_t q_write_ptr = get_write_ptr(cb_q_in);
        noc_async_read(q_read_addr, q_write_ptr, q_chunk_size_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_q_in, q_chunk_tiles);
    }

    // Create KV cache reader
    const auto k_reader = TensorAccessor(k_args, k_addr, k_tile_bytes);

    // Number of chunks per batch = max_seq_len / k_chunk_size = St / Sk_chunk_t
    constexpr uint32_t num_chunks_per_batch = St / Sk_chunk_t;

    // Set up multicast semaphore
    const uint32_t mcast_semaphore_addr = get_semaphore(mcast_semaphore_id);
    volatile tt_l1_ptr uint32_t* mcast_semaphore_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_semaphore_addr);

    // Set up NCRISC<->BRISC sync semaphore
    const uint32_t ncrisc_brisc_sync_l1_addr = get_semaphore(ncrisc_brisc_sync_semaphore_id);
    volatile tt_l1_ptr uint32_t* ncrisc_brisc_sync_curr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ncrisc_brisc_sync_l1_addr);
    volatile tt_l1_ptr uint32_t* ncrisc_brisc_sync_next_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ncrisc_brisc_sync_l1_addr + 4);
    volatile tt_l1_ptr uint32_t* k_write_curr_ptr_shared =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ncrisc_brisc_sync_l1_addr + 8);
    volatile tt_l1_ptr uint32_t* k_write_next_ptr_shared =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ncrisc_brisc_sync_l1_addr + 12);

    // Set up receiver_ready semaphore for double-buffer synchronization
    // Receivers signal sender when they've reserved CB space (ensuring consistent addresses)
    const uint32_t receiver_ready_semaphore_addr = get_semaphore(receiver_ready_semaphore_id);
    volatile tt_l1_ptr uint32_t* receiver_ready_semaphore_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_ready_semaphore_addr);
    // Sender's receiver_ready semaphore NOC address (mcast_start is sender's physical coords)
    const uint64_t sender_receiver_ready_noc_addr =
        get_noc_addr(mcast_start_x, mcast_start_y, receiver_ready_semaphore_addr);

    // Single KV head, single batch - no outer loop needed
    // kv_batch = 0 (Bkv = 1)
    constexpr uint32_t kv_batch = 0;

    if (is_mcast_sender) {
        // Program noc registers from first chunk
        const uint32_t shard_id = kv_batch * num_chunks_per_batch + k_chunk_start;
        uint64_t k_src_noc_addr = get_shard_noc_addr_helper(k_reader, shard_id);
        noc_async_read_one_packet_set_state<true>(k_src_noc_addr, k_page_size, vc);
    }

    // Strided iteration: core N processes chunks N, N+stride, N+2*stride, ...
    for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; k_chunk += num_cores_per_head) {
        {
            DeviceZoneScopedN("reader-k-read");

            // Step 1: All cores reserve CB space (determines write pointer)
            cb_reserve_back(cb_k_in, k_chunk_tiles);
            uint32_t k_write_ptr = get_write_ptr(cb_k_in);

            if (is_mcast_sender) {
                // Sender reads from ND-sharded DRAM
                // (Writer/BRISC will wait for receivers before multicast for double-buffer safety)
                DeviceZoneScopedN("mcast-sender-sharded-read");
                const uint32_t shard_id = kv_batch * num_chunks_per_batch + k_chunk;
                uint64_t k_src_noc_addr = get_shard_noc_addr_helper(k_reader, shard_id);

                constexpr uint32_t NUM_TRIDS = NOC_MAX_TRANSACTION_ID - 1;
                uint32_t src_base_addr = (uint32_t)(k_src_noc_addr & 0xFFFFFFFF);
                uint32_t src_offset = 0;
                uint32_t dst_addr = k_write_ptr;

                uint32_t curr_trid = 1;
                uint32_t wait_trid = 1;
                uint32_t pages_issued = 0;
                uint32_t pages_completed = 0;

                noc_semaphore_wait(ncrisc_brisc_sync_curr_ptr, 0);
                *k_write_curr_ptr_shared = k_write_ptr;
                for (uint32_t i = 0; i < NUM_TRIDS && pages_issued < k_num_pages; ++i) {
                    noc_async_read_set_trid(curr_trid);
                    noc_async_read_one_packet_with_state_with_trid(src_base_addr, src_offset, dst_addr, curr_trid);
                    src_offset += k_page_size;
                    dst_addr += k_page_size;
                    curr_trid = (curr_trid % NUM_TRIDS) + 1;
                    pages_issued++;
                }

                while (pages_completed < k_num_pages) {
                    noc_async_read_barrier_with_trid(wait_trid);
                    *ncrisc_brisc_sync_curr_ptr += 1;
                    pages_completed++;

                    if (pages_issued < k_num_pages) {
                        noc_async_read_set_trid(curr_trid);
                        noc_async_read_one_packet_with_state_with_trid(src_base_addr, src_offset, dst_addr, curr_trid);
                        src_offset += k_page_size;
                        dst_addr += k_page_size;
                        curr_trid = (curr_trid % NUM_TRIDS) + 1;
                        pages_issued++;
                    }

                    wait_trid = (wait_trid % NUM_TRIDS) + 1;
                }

                std::swap(ncrisc_brisc_sync_curr_ptr, ncrisc_brisc_sync_next_ptr);
                std::swap(k_write_curr_ptr_shared, k_write_next_ptr_shared);
            } else {
                // Step 2: Receiver signals sender that CB is reserved (address is ready)
                DeviceZoneScopedN("mcast-receiver-signal-ready");
                noc_semaphore_inc(sender_receiver_ready_noc_addr, 1);

                // Step 3: Receiver waits for multicast data
                noc_semaphore_wait(mcast_semaphore_ptr, MCAST_VALID);
                noc_semaphore_set(mcast_semaphore_ptr, MCAST_INVALID);
            }

            cb_push_back(cb_k_in, k_chunk_tiles);
        }
        // V is read directly from K buffer by compute kernel (strided matmul)
        // No need to copy V tiles - MLA optimization
    }
}
