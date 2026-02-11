// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"

#include "../rt_args_common.hpp"

// Compute semaphore increment and shift based on bits_per_step
// step_semaphore_inc[step] = 1 << (step * bits_per_step)
// step_semaphore_shift[step] = step * bits_per_step
// FORCE_INLINE ensures these are inlined even when called with runtime 'step'
template <uint32_t bits_per_step>
FORCE_INLINE constexpr uint32_t step_semaphore_inc(uint32_t step) {
    return 1U << (step * bits_per_step);
}
template <uint32_t bits_per_step>
FORCE_INLINE constexpr uint32_t step_semaphore_shift(uint32_t step) {
    return step * bits_per_step;
}

/******************************************************************************
 *                   Kernel Main                                               *
 ******************************************************************************/
void kernel_main() {
    /*
    Simplified Flash MLA Decode writer kernel for Deepseek V3 B1.
    Assumptions:
    - PNHt = 1 (single tile of Q heads per core)
    - num_kv_heads = 1 (MLA has single KV head)
    - num_heads_per_core = 1 (single head group per core)
    - KV cache is always ND-sharded
    - Output is always sharded
    */
    constexpr uint32_t vDHt = get_compile_time_arg_val(0);        // V head dim in tiles
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(1);  // tiles per K chunk
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(2);  // cores for seq len parallelism (8)
    uint32_t reducer_semaphore_addr = get_semaphore(get_compile_time_arg_val(3));
    constexpr uint32_t k_chunk_size = get_compile_time_arg_val(4);
    constexpr uint32_t q_tile_height = get_compile_time_arg_val(5);
    constexpr uint32_t DHt = get_compile_time_arg_val(6);
    constexpr uint32_t num_mcast_dests = get_compile_time_arg_val(7);
    constexpr uint32_t mcast_semaphore_id = get_compile_time_arg_val(8);
    constexpr uint32_t ncrisc_brisc_sync_semaphore_id = get_compile_time_arg_val(9);
    constexpr uint32_t k_page_size = get_compile_time_arg_val(10);
    constexpr uint32_t k_num_pages = get_compile_time_arg_val(11);
    constexpr uint32_t num_tree_reduction_steps = get_compile_time_arg_val(12);
    constexpr uint32_t receiver_ready_semaphore_id = get_compile_time_arg_val(13);
    constexpr uint32_t cb_index_id = get_compile_time_arg_val(14);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(15);
    constexpr uint32_t cb_ms_in = get_compile_time_arg_val(16);
    constexpr uint32_t cb_out_o = get_compile_time_arg_val(17);
    constexpr uint32_t cb_out_ms = get_compile_time_arg_val(18);
    constexpr uint32_t cb_intermed_out = get_compile_time_arg_val(19);

    uint32_t arg_idx = 0;
    const uint32_t cur_batch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const bool is_mcast_sender = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t mcast_start_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_start_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_end_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_end_y = get_arg_val<uint32_t>(arg_idx++);

    // Tree reduction info: 3 steps × 4 values (role, partner_s_block_idx, x, y)
    tt_l1_ptr uint32_t* tree_reduction_info = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_tree_reduction_steps * 4;

    // Get cur_pos from position tensor (MLA decode is always causal)
    uint32_t cur_pos;
    {
        cb_wait_front(cb_index_id, 1);
        uint32_t index_cb_ptr = get_read_ptr(cb_index_id);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_ptr);
        cur_pos = index_ptr[0];  // Single batch, position at index 0
    }

    // Sequence length assignment
    auto [k_num_chunks, k_chunk_start, k_chunk_end] =
        get_runtime_args(cur_pos, cur_batch, core_num_in_reduce, num_cores_per_head, k_chunk_size);

    if (k_chunk_start == k_chunk_end) {
        return;
    }

    // PNHt = 1, so out_chunk_tiles = vDHt
    constexpr uint32_t out_chunk_tiles = vDHt;

    // =========================================================================
    // KV Cache Multicast (page-level pipelining)
    // =========================================================================
    if (is_mcast_sender) {
        constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
        constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;

        const uint32_t mcast_semaphore_addr = get_semaphore(mcast_semaphore_id);
        volatile tt_l1_ptr uint32_t* mcast_semaphore_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_semaphore_addr);
        const uint32_t ncrisc_brisc_sync_addr = get_semaphore(ncrisc_brisc_sync_semaphore_id);
        volatile tt_l1_ptr uint32_t* ncrisc_brisc_sync_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ncrisc_brisc_sync_addr);
        volatile tt_l1_ptr uint32_t* k_write_ptr_shared =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ncrisc_brisc_sync_addr + 4);

        // Receiver ready semaphore: wait for all receivers to reserve CB before multicast
        // This ensures consistent write addresses across cores for double-buffer safety
        const uint32_t receiver_ready_semaphore_addr = get_semaphore(receiver_ready_semaphore_id);
        volatile tt_l1_ptr uint32_t* receiver_ready_semaphore_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_ready_semaphore_addr);

        constexpr uint8_t MCAST_NOC = 0;
        const uint64_t mcast_noc_addr =
            get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, 0);
        const uint64_t mcast_sem_addr = mcast_noc_addr | mcast_semaphore_addr;

        noc_semaphore_set(mcast_semaphore_ptr, 1);

        // Single head, strided iteration over chunks
        for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; k_chunk += num_cores_per_head) {
            DeviceZoneScopedN("mcast-sender-multicast");

            // Wait for NCRISC to have first page ready
            noc_semaphore_wait_min(ncrisc_brisc_sync_ptr, 1);
            invalidate_l1_cache();
            uint32_t k_write_ptr = *k_write_ptr_shared;

            // Wait for all receivers to have reserved their CB space before multicast
            // This allows DRAM reads to overlap with receiver CB reservation
            noc_semaphore_wait(receiver_ready_semaphore_ptr, num_mcast_dests);
            noc_semaphore_set(receiver_ready_semaphore_ptr, 0);  // Reset for next iteration

            // Page-level pipelining (KV cache is always sharded)
            uint32_t page_addr = k_write_ptr;
            uint64_t mcast_dest_addr = mcast_noc_addr | page_addr;
            noc_async_write_multicast(page_addr, mcast_dest_addr, k_page_size, num_mcast_dests, false, MCAST_NOC);

            for (uint32_t page = 1; page < k_num_pages; ++page) {
                noc_semaphore_wait_min(ncrisc_brisc_sync_ptr, page + 1);
                page_addr = k_write_ptr + page * k_page_size;
                mcast_dest_addr = mcast_noc_addr | page_addr;
                noc_async_write_multicast(page_addr, mcast_dest_addr, k_page_size, num_mcast_dests, false, MCAST_NOC);
            }

            noc_semaphore_set_multicast(mcast_semaphore_addr, mcast_sem_addr, num_mcast_dests, false, MCAST_NOC);
            noc_async_writes_flushed();
            *ncrisc_brisc_sync_ptr = 0;
        }
    }

    // =========================================================================
    // Tree Reduction
    // =========================================================================
    constexpr uint32_t tile_bytes_intermed = get_tile_size(cb_intermed_out);
    constexpr uint32_t o_write_size = out_chunk_tiles * tile_bytes_intermed;
    // PNHt = 1, m and s packed into single tile
    constexpr uint32_t ms_write_size = tile_bytes_intermed;
    constexpr uint32_t per_step_buffer_size = o_write_size + ms_write_size;

    constexpr uint32_t bits_per_step = 1;
    constexpr uint32_t step_mask = (1U << bits_per_step) - 1;

    volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reducer_semaphore_addr);

    bool needs_reduction = (k_chunk_end - k_chunk_start < k_num_chunks);
    uint32_t num_active_s_blocks = (k_num_chunks < num_cores_per_head) ? k_num_chunks : num_cores_per_head;

    if (needs_reduction) {
        for (uint32_t step = 0; step < num_tree_reduction_steps; ++step) {
            DeviceZoneScopedN("tree-reduction-step");
            uint32_t role_code = tree_reduction_info[step * 4 + 0];
            uint32_t partner_s_block_idx = tree_reduction_info[step * 4 + 1];
            uint32_t partner_x = tree_reduction_info[step * 4 + 2];
            uint32_t partner_y = tree_reduction_info[step * 4 + 3];

            if (role_code != 0 && partner_s_block_idx >= num_active_s_blocks) {
                continue;
            }

            uint32_t step_buffer_offset = step * per_step_buffer_size;

            if (role_code == 1) {
                // SENDER
                DeviceZoneScopedN("tree-reduction-sender");

                cb_wait_front(cb_out_o, out_chunk_tiles);
                cb_wait_front(cb_out_ms, 1);  // m and s packed into single tile

                uint64_t output_write_addr =
                    get_noc_addr(partner_x, partner_y, get_write_ptr(cb_intermed_out) + step_buffer_offset);

                // Write m/s (packed in single tile)
                noc_async_write<NOC_MAX_BURST_SIZE + 1, false, /*posted=*/true>(
                    get_read_ptr(cb_out_ms), output_write_addr, ms_write_size);
                output_write_addr += ms_write_size;

                // Write O
                noc_async_write<NOC_MAX_BURST_SIZE + 1, false, /*posted=*/true>(
                    get_read_ptr(cb_out_o), output_write_addr, o_write_size);

                uint64_t partner_semaphore_addr = get_noc_addr(partner_x, partner_y, reducer_semaphore_addr);
                noc_semaphore_inc(partner_semaphore_addr, step_semaphore_inc<bits_per_step>(step));

                noc_async_posted_writes_flushed();
                cb_pop_front(cb_out_ms, 1);
                cb_pop_front(cb_out_o, out_chunk_tiles);
                noc_async_atomic_barrier();
                return;

            } else if (role_code == 2) {
                // RECEIVER
                DeviceZoneScopedN("tree-reduction-receiver");

                while (true) {
                    invalidate_l1_cache();
                    uint32_t sem_val = *in0_receiver_semaphore_addr_ptr;
                    uint8_t step_sem = (sem_val >> step_semaphore_shift<bits_per_step>(step)) & step_mask;
                    if (step_sem >= 1) {
                        break;
                    }
                }

                // TODO: Get rid of the intermediate CB and copy
                uint64_t intermed_l1_read_addr = get_noc_addr(get_read_ptr(cb_intermed_out) + step_buffer_offset);

                // Read m/s (packed in single tile)
                cb_reserve_back(cb_ms_in, 1);
                noc_async_read(intermed_l1_read_addr, get_read_ptr(cb_ms_in), ms_write_size);
                intermed_l1_read_addr += ms_write_size;
                noc_async_read_barrier();
                cb_push_back(cb_ms_in, 1);

                // Read O
                cb_reserve_back(cb_out_o, out_chunk_tiles);
                noc_async_read(intermed_l1_read_addr, get_read_ptr(cb_out_o), o_write_size);
                noc_async_read_barrier();
                cb_push_back(cb_out_o, out_chunk_tiles);
            }
        }
    }

    noc_async_write_barrier();
}
