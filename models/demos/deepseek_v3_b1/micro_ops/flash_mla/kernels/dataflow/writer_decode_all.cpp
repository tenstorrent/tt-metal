// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "api/debug/assert.h"

#include "../rt_args_common.hpp"

/******************************************************************************
 *                   Helper Functions                                          *
 ******************************************************************************/
template <uint32_t tile_bytes>
void copy_tile(uint64_t noc_read_addr_base, uint32_t q_write_ptr_base, uint32_t src_tile_id, uint32_t dst_tile_id) {
    noc_async_read(
        noc_read_addr_base + src_tile_id * tile_bytes, q_write_ptr_base + dst_tile_id * tile_bytes, tile_bytes);
}

template <uint32_t tile_bytes>
void fill_tile(uint32_t cb_id, uint32_t tile_id, uint32_t val) {
    if (val == 0) {
        constexpr uint32_t num_zeros_reads = tile_bytes / MEM_ZEROS_SIZE;
        uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
        uint32_t write_addr = get_write_ptr(cb_id) + tile_id * tile_bytes;
        volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);

        // Fill tile with zeros
        for (uint32_t i = 0; i < num_zeros_reads; ++i) {
            noc_async_read(zeros_noc_addr, write_addr, MEM_ZEROS_SIZE);
            write_addr += MEM_ZEROS_SIZE;
        }
        noc_async_read_barrier();
    } else {
        // Fill 2 uint16 datums in each writes to optimize for performance
        volatile tt_l1_ptr uint32_t* ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id) + tile_id * tile_bytes);
        constexpr int num_uint32_datums_tile = tile_bytes / 4;
        for (int k = 0; k < num_uint32_datums_tile; k++) {
            ptr[k] = val;
        }
    }
}

template <uint32_t tile_bytes, uint32_t tile_height>
void fill_tile_partial(uint32_t cb_id, uint32_t tile_id, uint32_t cur_pos_in_tile, uint32_t partial_val) {
    /*
    We want to fill columns > cur_pos_in_tile with partial_val (-inf for masking).
    cur_pos_in_tile is the column position (0-31 range for 32-column tiles).

    Tile face layout:
    - 32x32 tile: 4 faces (2 row halves × 2 col halves), each 16x16
    - 16x32 tile: 2 faces (1 row half × 2 col halves), each 16x16
    - 8x32 tile:  2 faces (1 row half × 2 col halves), each 8x16
    */
    // Number of faces: 2 for tile_height <= 16, 4 for tile_height > 16
    constexpr int num_faces = (tile_height <= 16) ? 2 : 4;
    constexpr uint32_t num_rows_in_face = (tile_height <= 16) ? tile_height : 16;
    constexpr uint32_t num_cols_in_face = 16;
    constexpr uint32_t face_size_uint16 = num_rows_in_face * num_cols_in_face;
    constexpr uint32_t face_size_uint32 = face_size_uint16 / 2;

    fill_tile<tile_bytes>(cb_id, tile_id, 0);
    if (cur_pos_in_tile == 31 || partial_val == 0) {
        return;
    }
    const uint16_t datum_val = partial_val >> 16;
    volatile tt_l1_ptr uint16_t* uint16_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id) + tile_id * tile_bytes);
    volatile tt_l1_ptr uint32_t* uint32_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id) + tile_id * tile_bytes);

    // Determine which column half (face) to start from
    // face_start = 0 for cols 0-15 (left half), face_start = 1 for cols 16-31 (right half)
    int face_start = (cur_pos_in_tile < 15) ? 0 : 1;
    uint32_t fill_pos_in_face = (cur_pos_in_tile + 1) % 16;

    if (face_start == 0) {
        // Fill the entire right column half (faces 1, 3, ...) with partial_val
        constexpr int num_uint32_datums_per_face = face_size_uint32;
        for (int k = 1; k < num_faces; k += 2) {
            uint32_t uint32_face_idx = k * face_size_uint32;
            for (int j = 0; j < num_uint32_datums_per_face; j++) {
                uint32_ptr[uint32_face_idx + j] = partial_val;
            }
        }
    }

    // Fill partial columns in the current face(s)
    // Optimizing by filling 2 uint16 datums per write where possible
    bool is_odd_pos_filled = fill_pos_in_face % 2 == 1;
    uint32_t fill_pos_in_uint32_face = (fill_pos_in_face + 1) >> 1;
    constexpr uint32_t num_cols_in_uint32_face = num_cols_in_face >> 1;

    for (int k = face_start; k < num_faces; k += 2) {
        uint32_t uint16_face_idx = k * face_size_uint16;
        uint32_t uint32_face_idx = k * face_size_uint32;

        for (uint32_t face_row_idx = 0; face_row_idx < num_rows_in_face; face_row_idx++) {
            // If fill_pos_in_face is odd, fill that single position first
            if (is_odd_pos_filled) {
                uint16_ptr[uint16_face_idx + (fill_pos_in_face + num_cols_in_face * face_row_idx)] = datum_val;
            }

            // Fill remaining positions in pairs (uint32)
            for (uint32_t uint32_face_col_idx = fill_pos_in_uint32_face; uint32_face_col_idx < num_cols_in_uint32_face;
                 uint32_face_col_idx++) {
                uint32_ptr[uint32_face_idx + (uint32_face_col_idx + num_cols_in_uint32_face * face_row_idx)] =
                    partial_val;
            }
        }
    }
}

template <uint32_t cb_mask_in, uint32_t q_tile_height>
void generate_mask(uint32_t k_num_chunks, uint32_t Sk_chunk_t, uint32_t cur_pos) {
    /*
    Generate causal mask for MLA decode.
    Positions <= cur_pos are allowed (0), positions > cur_pos are masked (-inf).
    PNHt = 1 simplification: only generate Sk_chunk_t tiles.
    */

    // PNHt = 1, so total_read_tiles = Sk_chunk_t
    uint32_t total_read_tiles = Sk_chunk_t;
    uint32_t cur_pos_in_chunk = cur_pos % (Sk_chunk_t * 32);
    uint32_t cur_pos_in_chunk_t = cur_pos_in_chunk / 32;
    uint32_t cur_pos_in_tile = cur_pos_in_chunk % 32;
    constexpr uint32_t NEG_INF = 0xFF80FF80;

    cb_reserve_back(cb_mask_in, total_read_tiles);

    uint64_t noc_read_addr_base = get_noc_addr(get_read_ptr(cb_mask_in));
    uint32_t q_write_ptr_base = get_read_ptr(cb_mask_in);
    constexpr uint32_t tile_bytes = get_tile_size(cb_mask_in);

    for (uint32_t i = 0; i < Sk_chunk_t; ++i) {
        if (i < cur_pos_in_chunk_t) {
            // fill with zero
            if (i == 0) {
                fill_tile<tile_bytes>(cb_mask_in, i, 0);
            } else {
                copy_tile<tile_bytes>(noc_read_addr_base, q_write_ptr_base, 0, i);
                if (i == cur_pos_in_chunk_t - 1) {
                    noc_async_read_barrier();
                }
            }
        } else if (i == cur_pos_in_chunk_t) {
            // fill with partial zero/-inf
            fill_tile_partial<tile_bytes, q_tile_height>(cb_mask_in, i, cur_pos_in_tile, NEG_INF);
        } else {
            // fill with -inf
            if (i == cur_pos_in_chunk_t + 1) {
                fill_tile<tile_bytes>(cb_mask_in, i, NEG_INF);
            } else {
                copy_tile<tile_bytes>(noc_read_addr_base, q_write_ptr_base, cur_pos_in_chunk_t + 1, i);
                if (i == Sk_chunk_t - 1) {
                    noc_async_read_barrier();
                }
            }
        }
        // PNHt = 1, so no inner loop needed
    }

    cb_push_back(cb_mask_in, total_read_tiles);
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
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(2);
    constexpr uint32_t zero_scalar_packed = get_compile_time_arg_val(3);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(4);  // cores for seq len parallelism (8)
    uint32_t reducer_semaphore_addr = get_semaphore(get_compile_time_arg_val(5));
    constexpr uint32_t k_chunk_size = get_compile_time_arg_val(6);
    constexpr uint32_t q_tile_height = get_compile_time_arg_val(7);
    constexpr uint32_t DHt = get_compile_time_arg_val(8);
    constexpr uint32_t num_mcast_dests = get_compile_time_arg_val(9);
    constexpr uint32_t mcast_semaphore_id = get_compile_time_arg_val(10);
    constexpr uint32_t ncrisc_brisc_sync_semaphore_id = get_compile_time_arg_val(11);
    constexpr uint32_t k_page_size = get_compile_time_arg_val(12);
    constexpr uint32_t k_num_pages = get_compile_time_arg_val(13);
    constexpr uint32_t num_tree_reduction_steps = get_compile_time_arg_val(14);
    constexpr uint32_t receiver_ready_semaphore_id = get_compile_time_arg_val(15);

    uint32_t arg_idx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
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
        constexpr uint32_t cb_index_id = tt::CBIndex::c_8;
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

    constexpr uint32_t cb_intermed_out = tt::CBIndex::c_19;
    constexpr uint32_t cb_out_o = tt::CBIndex::c_16;
    constexpr uint32_t cb_m_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_l_in = tt::CBIndex::c_7;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_11;
    constexpr uint32_t cb_zero_in = tt::CBIndex::c_12;
    constexpr uint32_t cb_out_worker = tt::CBIndex::c_16;
    constexpr uint32_t cb_out_m = tt::CBIndex::c_17;
    constexpr uint32_t cb_out_l = tt::CBIndex::c_18;

    // Generate scalers
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);
    generate_reduce_scaler(cb_zero_in, zero_scalar_packed);
    generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed);

    // =========================================================================
    // KV Cache Multicast (page-level pipelining)
    // =========================================================================
    if (is_mcast_sender) {
        constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
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

            noc_async_write_barrier();
            noc_semaphore_set_multicast(mcast_semaphore_addr, mcast_sem_addr, num_mcast_dests, false, MCAST_NOC);
            *ncrisc_brisc_sync_ptr = 0;
        }
    }

    // Generate causal mask (PNHt = 1 simplification)
    generate_mask<cb_mask_in, q_tile_height>(k_num_chunks, Sk_chunk_t, cur_pos);

    noc_async_write_barrier();

    // =========================================================================
    // Tree Reduction
    // =========================================================================
    constexpr uint32_t tile_bytes_intermed = get_tile_size(cb_intermed_out);
    constexpr uint32_t o_write_size = out_chunk_tiles * tile_bytes_intermed;
    // PNHt = 1, so ml_write_size = tile_bytes_intermed
    constexpr uint32_t ml_write_size = tile_bytes_intermed;
    constexpr uint32_t per_step_buffer_size = o_write_size + 2 * ml_write_size;

    constexpr uint32_t step_semaphore_inc[3] = {1, 256, 65536};
    constexpr uint32_t step_semaphore_shift[3] = {0, 8, 16};

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

                cb_wait_front(cb_out_worker, out_chunk_tiles);
                cb_wait_front(cb_out_m, 1);  // PNHt = 1
                cb_wait_front(cb_out_l, 1);

                uint64_t output_write_addr =
                    get_noc_addr(partner_x, partner_y, get_write_ptr(cb_intermed_out) + step_buffer_offset);

                noc_async_write(get_read_ptr(cb_out_m), output_write_addr, ml_write_size);
                output_write_addr += ml_write_size;
                noc_async_write(get_read_ptr(cb_out_l), output_write_addr, ml_write_size);
                output_write_addr += ml_write_size;
                noc_async_write(get_read_ptr(cb_out_worker), output_write_addr, o_write_size);

                noc_async_write_barrier();
                uint64_t partner_semaphore_addr = get_noc_addr(partner_x, partner_y, reducer_semaphore_addr);
                noc_semaphore_inc(partner_semaphore_addr, step_semaphore_inc[step]);

                cb_pop_front(cb_out_worker, out_chunk_tiles);
                cb_pop_front(cb_out_m, 1);
                cb_pop_front(cb_out_l, 1);

                noc_async_atomic_barrier();
                return;

            } else if (role_code == 2) {
                // RECEIVER
                DeviceZoneScopedN("tree-reduction-receiver");

                while (true) {
                    invalidate_l1_cache();
                    uint32_t sem_val = *in0_receiver_semaphore_addr_ptr;
                    uint8_t step_sem = (sem_val >> step_semaphore_shift[step]) & 0xFF;
                    if (step_sem >= 1) {
                        break;
                    }
                }

                uint64_t intermed_l1_read_addr = get_noc_addr(get_read_ptr(cb_intermed_out) + step_buffer_offset);

                cb_reserve_back(cb_m_in, 1);
                noc_async_read(intermed_l1_read_addr, get_read_ptr(cb_m_in), ml_write_size);
                intermed_l1_read_addr += ml_write_size;
                noc_async_read_barrier();
                cb_push_back(cb_m_in, 1);

                cb_reserve_back(cb_l_in, 1);
                noc_async_read(intermed_l1_read_addr, get_read_ptr(cb_l_in), ml_write_size);
                intermed_l1_read_addr += ml_write_size;
                noc_async_read_barrier();
                cb_push_back(cb_l_in, 1);

                cb_reserve_back(cb_out_o, out_chunk_tiles);
                noc_async_read(intermed_l1_read_addr, get_read_ptr(cb_out_o), o_write_size);
                noc_async_read_barrier();
                cb_push_back(cb_out_o, out_chunk_tiles);
            }
        }
    }

    noc_async_writes_flushed();
}
