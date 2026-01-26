// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <vector>

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
    Simplified Flash MLA Decode reader kernel.
    Q is always sharded, KV cache can be DRAM interleaved or HEIGHT_SHARDED.
    For HEIGHT_SHARDED KV cache, each shard = one K chunk (k_chunk_size x kvpe_dim).
    */
    constexpr uint32_t B = get_compile_time_arg_val(0);           // batch size
    constexpr uint32_t PNHt = get_compile_time_arg_val(1);        // padded number of heads in tiles
    constexpr uint32_t St = get_compile_time_arg_val(2);          // full sequence length of kv cache in tiles
    constexpr uint32_t DHt = get_compile_time_arg_val(3);         // head dim
    constexpr uint32_t vDHt = get_compile_time_arg_val(4);        // head dim of V
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(5);  // number of tiles in seqlen of a k/v/mask chunk
    constexpr uint32_t num_cores = get_compile_time_arg_val(6);
    constexpr uint32_t num_cores_per_batch = get_compile_time_arg_val(7);
    constexpr uint32_t k_chunk_size = get_compile_time_arg_val(8);
    constexpr uint32_t index_stick_size_B = get_compile_time_arg_val(9);
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(10);
    constexpr uint32_t Bkv = get_compile_time_arg_val(11);
    constexpr uint32_t q_heads_parallel_factor = get_compile_time_arg_val(12);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(13);
    constexpr uint32_t num_heads_per_core = get_compile_time_arg_val(14);
    constexpr uint32_t num_output_cores = get_compile_time_arg_val(15);
    constexpr uint32_t max_dynamic_chunk_size = get_compile_time_arg_val(16);
    constexpr bool tilize_q = get_compile_time_arg_val(17) == 1;
    constexpr uint32_t q_chunk_size_bytes = get_compile_time_arg_val(18);
    constexpr uint32_t num_mcast_dests = get_compile_time_arg_val(19);
    constexpr uint32_t mcast_semaphore_id = get_compile_time_arg_val(20);

    // TensorAccessorArgs for K and V (KV cache in DRAM), and pos tensor
    constexpr auto k_args = TensorAccessorArgs<21>();  // After compile-time args
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    uint32_t arg_idx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pos_addr = get_arg_val<uint32_t>(arg_idx++);
    const bool is_worker = get_arg_val<uint32_t>(arg_idx++) == 0;
    const bool is_output_core = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t cur_head_group = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_batch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_output = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_pos_arg = get_arg_val<uint32_t>(arg_idx++);
    const bool is_mcast_sender = get_arg_val<uint32_t>(arg_idx++) == 1;
    // Multicast coordinates (physical NOC coords for this core's S block bounding box)
    const uint32_t mcast_start_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_start_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_end_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_end_y = get_arg_val<uint32_t>(arg_idx++);

    // idle core
    if (q_addr == 0) {
        return;
    }

    // Get cur_pos (MLA decode is always causal)
    uint32_t cur_pos;
    // using UINT32_MAX as a flag to indicate that cur_pos is not provided as a list
    if (cur_pos_arg != UINT32_MAX) {
        cur_pos = cur_pos_arg;
    } else {
        constexpr uint32_t cb_index_id = tt::CBIndex::c_8;
        cb_reserve_back(cb_index_id, 1);
        uint32_t index_cb_wr_ptr = get_write_ptr(cb_index_id);

        // Read cur_pos tensor from DRAM interleaved
        const auto pos_reader = TensorAccessor(pos_args, pos_addr, index_stick_size_B);
        uint64_t tensor_index_noc_addr = pos_reader.get_noc_addr(0);
        noc_async_read(tensor_index_noc_addr, index_cb_wr_ptr, index_stick_size_B);
        noc_async_read_barrier();

        cb_push_back(cb_index_id, 1);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_wr_ptr);
        cur_pos = index_ptr[cur_batch / q_heads_parallel_factor];
    }

    if (cur_pos == UINT32_MAX) {
        // cur_pos of -1 indicates that the user should be skipped
        return;
    }

    auto Sk_chunk_t_dynamic = get_dynamic_Sk_chunk_t<Sk_chunk_t, max_dynamic_chunk_size>(cur_pos);
    auto k_chunk_size_dynamic = Sk_chunk_t_dynamic * tt::constants::TILE_HEIGHT;

    // Sequence length assignment (no sliding window for MLA)
    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk] = get_runtime_args(
        cur_pos, cur_batch, core_num_in_reduce, num_cores_per_head, k_chunk_size_dynamic, std::nullopt);

    if (k_chunk_start == k_chunk_end) {
        return;  // early exit because no computes needs to be done
    }

    tt_l1_ptr uint32_t* all_output_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_output_cores;
    tt_l1_ptr uint32_t* all_output_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx++));

    uint32_t output_core_noc_x = all_output_noc_x[cur_batch];
    uint32_t output_core_noc_y = all_output_noc_y[cur_batch];

    constexpr uint32_t q_chunk_tiles = PNHt * DHt;
    uint32_t k_chunk_tiles = Sk_chunk_t_dynamic * DHt;
    uint32_t v_chunk_tiles = Sk_chunk_t_dynamic * vDHt;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_q_rm = tt::CBIndex::c_10;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);

    // Read Q from sharded memory (Q is always sharded)
    {
        DeviceZoneScopedN("reader-q-read");
        uint64_t q_read_addr;
        uint32_t q_write_ptr;
        if (is_output_core) {
            q_read_addr = get_noc_addr(q_addr);
        } else {
            q_read_addr = get_noc_addr(output_core_noc_x, output_core_noc_y, q_addr);
        }
        if constexpr (tilize_q) {
            cb_reserve_back(cb_q_rm, q_chunk_tiles);
            q_write_ptr = get_write_ptr(cb_q_rm);
        } else {
            cb_reserve_back(cb_q_in, q_chunk_tiles);
            q_write_ptr = get_write_ptr(cb_q_in);
        }
        // Q tensor is properly set up with tiny tiles, just read contiguously
        noc_async_read(q_read_addr, q_write_ptr, q_chunk_size_bytes);
        noc_async_read_barrier();
        // DPRINT << TileSlice(cb_q_in, 0, SliceRange{.h0 = 0, .h1 = 8, .hs = 1, .w0 = 0, .w1 = 32, .ws = 8}, true,
        // true) << ENDL();
        if constexpr (tilize_q) {
            cb_push_back(cb_q_rm, q_chunk_tiles);
        } else {
            cb_push_back(cb_q_in, q_chunk_tiles);
        }
    }

    // Create KV cache reader (only used by mcast sender)
    const auto k_reader = TensorAccessor(k_args, k_addr, k_tile_bytes);

    // Number of chunks per batch = max_seq_len / k_chunk_size = St / Sk_chunk_t
    constexpr uint32_t num_chunks_per_batch = St / Sk_chunk_t;

    // Set up multicast addresses and semaphore
    const uint64_t mcast_noc_addr = get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, 0);
    const uint32_t mcast_semaphore_addr = get_semaphore(mcast_semaphore_id);
    volatile tt_l1_ptr uint32_t* mcast_semaphore_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_semaphore_addr);

    // Sender: set local semaphore to valid once (will be multicast each iteration)
    if (is_mcast_sender) {
        noc_semaphore_set(mcast_semaphore_ptr, MCAST_VALID);
    }

    for (uint32_t cur_head = cur_head_group * num_heads_per_core;
         cur_head < cur_head_group * num_heads_per_core + num_heads_per_core;
         ++cur_head) {
        // Batch index for KV cache
        const uint32_t kv_batch = (cur_batch / q_heads_parallel_factor) % Bkv;

        for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
            // Read K chunk in natural [Sk_chunk, DHt] order (no transpose)
            // cb_matmul_blocks with transpose=true handles [N, K] layout
            uint64_t k_base_read_ptr;
            {
                DeviceZoneScopedN("reader-k-read");
                cb_reserve_back(cb_k_in, k_chunk_tiles);
                uint32_t k_write_ptr = get_write_ptr(cb_k_in);
                k_base_read_ptr = get_noc_addr(k_write_ptr);

                const uint32_t k_chunk_bytes = k_chunk_tiles * k_tile_bytes;

                if (is_mcast_sender) {
                    // Sender: read from DRAM
                    if constexpr (k_args.is_sharded) {
                        DeviceZoneScopedN("mcast-sender-sharded-read");
                        {
                            DeviceZoneScopedN("mcast-sender-tensor-accessor");
                            const uint32_t shard_id = kv_batch * num_chunks_per_batch + k_chunk;
                            uint64_t k_src_noc_addr = get_shard_noc_addr_helper(k_reader, shard_id);
                            uint32_t dram_x = NOC_UNICAST_ADDR_X(k_src_noc_addr);
                            uint32_t dram_y = NOC_UNICAST_ADDR_Y(k_src_noc_addr);
                            DPRINT << "shard_id=" << shard_id << " -> DRAM(" << dram_x << "," << dram_y << ")"
                                   << ENDL();
                            noc_async_read(k_src_noc_addr, k_write_ptr, k_chunk_bytes);
                        }
                        noc_async_read_barrier();
                    } else {
                        DeviceZoneScopedN("mcast-sender-interleaved-read");
                        const uint32_t k_batch_offset = kv_batch * num_kv_heads * St * DHt;
                        const uint32_t k_head_offset = cur_head * St * DHt;
                        const uint32_t k_chunk_offset = k_chunk * Sk_chunk_t_dynamic * DHt;
                        uint32_t k_tile_id = k_batch_offset + k_head_offset + k_chunk_offset;
                        uint32_t write_ptr = k_write_ptr;
                        for (uint32_t tile = 0; tile < k_chunk_tiles; ++tile) {
                            noc_async_read_tile(k_tile_id, k_reader, write_ptr);
                            k_tile_id++;
                            write_ptr += k_tile_bytes;
                        }
                        noc_async_read_barrier();
                    }

                    // Multicast K data to other cores in the S block
                    {
                        DeviceZoneScopedN("mcast-sender-multicast");
                        // Multicast to other cores (sender already has data from DRAM read, no loopback needed)
                        uint64_t mcast_dest_addr = mcast_noc_addr | k_write_ptr;
                        noc_async_write_multicast(k_write_ptr, mcast_dest_addr, k_chunk_bytes, num_mcast_dests, true);

                        // Signal receivers that data is ready via multicast semaphore
                        uint64_t mcast_sem_addr = mcast_noc_addr | mcast_semaphore_addr;
                        noc_semaphore_set_multicast(mcast_semaphore_addr, mcast_sem_addr, num_mcast_dests);
                        noc_async_write_barrier();
                    }
                } else {
                    // Receiver: wait for multicast data from sender
                    DeviceZoneScopedN("mcast-receiver-wait");
                    noc_semaphore_wait(mcast_semaphore_ptr, MCAST_VALID);
                    noc_semaphore_set(mcast_semaphore_ptr, MCAST_INVALID);
                }

                cb_push_back(cb_k_in, k_chunk_tiles);
            }

            // Read V chunk from K's L1 buffer (MLA reuses K for V)
            // K is now stored as [Sk_chunk, DHt], so K[row, col] = row * DHt + col
            {
                DeviceZoneScopedN("reader-v-read");
                cb_reserve_back(cb_v_in, v_chunk_tiles);
                uint32_t v_write_ptr = get_write_ptr(cb_v_in);
                for (uint32_t row = 0; row < Sk_chunk_t_dynamic; ++row) {
                    uint64_t k_read_ptr = k_base_read_ptr + row * DHt * k_tile_bytes;  // Start of row
                    for (uint32_t col = 0; col < vDHt; ++col) {
                        noc_async_read(k_read_ptr, v_write_ptr, v_tile_bytes);
                        v_write_ptr += v_tile_bytes;
                        k_read_ptr += k_tile_bytes;  // Next column (stride 1)
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_v_in, v_chunk_tiles);
            }
        }
    }
}
