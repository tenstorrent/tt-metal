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

template <uint32_t cb_mask_in, uint32_t PNHt, uint32_t q_tile_height>
void generate_mask(uint32_t k_num_chunks, uint32_t Sk_chunk_t, uint32_t cur_pos) {
    /*
    Generate causal mask for MLA decode.
    Positions <= cur_pos are allowed (0), positions > cur_pos are masked (-inf).

    The mask tiles have shape (q_tile_height x 32) - same height as Q tiles, 32 columns for K positions.
    cur_pos is the sequence position in K, determining which columns to mask.
    */

    // the cb_mask in is of size PNHt * Sk_chunk_t
    uint32_t total_read_tiles = PNHt * Sk_chunk_t;
    uint32_t cur_pos_in_chunk = cur_pos % (Sk_chunk_t * 32);  // Column position within chunk (K uses 32-col tiles)
    uint32_t cur_pos_in_chunk_t = cur_pos_in_chunk / 32;      // Which tile column we're in
    uint32_t cur_pos_in_tile = cur_pos_in_chunk % 32;         // Column position within the tile
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
                copy_tile<tile_bytes>(
                    noc_read_addr_base, q_write_ptr_base, 0, i);  // copy from cb_mask_in[0] to cb_mask_in[i]
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
                copy_tile<tile_bytes>(
                    noc_read_addr_base,
                    q_write_ptr_base,
                    cur_pos_in_chunk_t + 1,
                    i);  // copy from cb_mask_in[cur_pos_in_chunk_t+1] to cb_mask_in[i]
                if (i == Sk_chunk_t - 1) {
                    noc_async_read_barrier();
                }
            }
        }
        for (uint32_t j = 1; j < PNHt; ++j) {
            // copy from cb_mask_in[i] to cb_mask_in[j*Sk_chunk_t + i]
            copy_tile<tile_bytes>(noc_read_addr_base, q_write_ptr_base, i, j * Sk_chunk_t + i);
            if (j == PNHt - 1) {
                noc_async_read_barrier();
            }
        }
    }

    cb_push_back(cb_mask_in, total_read_tiles);
}

template <
    uint32_t out_chunk_tiles,
    uint32_t cb_out,
    uint32_t cb_out_m,
    uint32_t cb_out_l,
    uint32_t cb_intermed_out,
    uint32_t PNHt>
void worker_compute(
    uint64_t in0_sender_semaphore_noc_addr,
    uint32_t worker_id,
    uint32_t reduce_core_noc_x,
    uint32_t reduce_core_noc_y) {
    uint32_t out_tile_id = 0;

    // Wait for compute to deliver output chunk
    cb_wait_front(cb_out, out_chunk_tiles);
    cb_wait_front(cb_out_m, PNHt);
    cb_wait_front(cb_out_l, PNHt);

    // Write output chunk to reducer
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    uint32_t worker_offset = worker_id * (out_chunk_tiles + 2 * PNHt) * tile_bytes;
    constexpr uint32_t o_write_size = out_chunk_tiles * tile_bytes;
    constexpr uint32_t ml_write_size = PNHt * tile_bytes;
    uint64_t output_write_addr =
        get_noc_addr(reduce_core_noc_x, reduce_core_noc_y, get_write_ptr(cb_intermed_out)) + worker_offset;

    // send the max logits first then the logits sum then the partial output to the reducer
    noc_async_write(get_read_ptr(cb_out_m), output_write_addr, ml_write_size);
    output_write_addr += ml_write_size;
    noc_async_write(get_read_ptr(cb_out_l), output_write_addr, ml_write_size);
    output_write_addr += ml_write_size;
    noc_async_write(get_read_ptr(cb_out), output_write_addr, o_write_size);

    // increment semaphore
    noc_async_write_barrier();
    noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);

    // pop front
    cb_pop_front(cb_out, out_chunk_tiles);
    cb_pop_front(cb_out_m, PNHt);
    cb_pop_front(cb_out_l, PNHt);
}

/******************************************************************************
 *                   Kernel Main                                               *
 ******************************************************************************/
void kernel_main() {
    /*
    Simplified Flash MLA Decode writer kernel.
    Output is always sharded. num_kv_heads is always 1 for MLA.
    */
    constexpr uint32_t B = get_compile_time_arg_val(0);           // batch size
    constexpr uint32_t PNHt = get_compile_time_arg_val(1);        // padded number of heads in tiles
    constexpr uint32_t St = get_compile_time_arg_val(2);          // full sequence length of kv cache in tiles
    constexpr uint32_t vDHt = get_compile_time_arg_val(3);        // head dim for V
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);  // number of tiles in seqlen of a k/v/mask chunk
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(5);
    constexpr uint32_t zero_scalar_packed = get_compile_time_arg_val(6);
    constexpr uint32_t scale_val = get_compile_time_arg_val(7);
    constexpr uint32_t num_cores_per_batch = get_compile_time_arg_val(8);           // num cores per batch
    constexpr uint32_t num_cores = get_compile_time_arg_val(9);                     // num running cores in total
    uint32_t reducer_semaphore_addr = get_semaphore(get_compile_time_arg_val(10));  // semaphore for reducer
    constexpr uint32_t k_chunk_size = get_compile_time_arg_val(11);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(12);
    constexpr uint32_t num_heads_per_core = get_compile_time_arg_val(13);
    constexpr uint32_t num_reducer_cores = get_compile_time_arg_val(14);
    constexpr uint32_t max_dynamic_chunk_size = get_compile_time_arg_val(15);
    constexpr uint32_t q_heads_parallel_factor = get_compile_time_arg_val(16);
    constexpr uint32_t q_tile_height = get_compile_time_arg_val(17);  // Q tile height for tiny tile support
    // Multicast args for DRAM/mcast overlap
    constexpr uint32_t DHt = get_compile_time_arg_val(18);              // head dim in tiles
    constexpr uint32_t num_mcast_dests = get_compile_time_arg_val(19);  // multicast destinations
    constexpr uint32_t mcast_semaphore_id = get_compile_time_arg_val(20);
    constexpr uint32_t ncrisc_brisc_sync_semaphore_id = get_compile_time_arg_val(21);
    constexpr uint32_t k_page_size = get_compile_time_arg_val(22);     // page size for pipelining
    constexpr uint32_t k_num_pages = get_compile_time_arg_val(23);     // pages per K chunk
    constexpr bool kv_is_sharded = get_compile_time_arg_val(24) == 1;  // page-level vs chunk-level

    uint32_t arg_idx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t worker_id_for_reduce = get_arg_val<uint32_t>(arg_idx++);
    const bool is_worker = get_arg_val<uint32_t>(arg_idx++) == 0;
    const uint32_t cur_head_group = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_batch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_pos_arg = get_arg_val<uint32_t>(arg_idx++);
    // Multicast args for DRAM/mcast overlap
    const bool is_mcast_sender = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t mcast_start_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_start_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_end_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_end_y = get_arg_val<uint32_t>(arg_idx++);

    // idle core
    if (out_addr == 0) {
        return;
    }

    // Get cur_pos (MLA decode is always causal)
    uint32_t cur_pos;
    // using UINT32_MAX as a flag to indicate that cur_pos is not provided as a list
    if (cur_pos_arg != UINT32_MAX) {
        cur_pos = cur_pos_arg;
    } else {
        constexpr uint32_t cb_index_id = tt::CBIndex::c_8;
        cb_wait_front(cb_index_id, 1);
        uint32_t index_cb_ptr = get_read_ptr(cb_index_id);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_ptr);
        cur_pos = index_ptr[(uint32_t)(cur_batch / q_heads_parallel_factor)];
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

    tt_l1_ptr uint32_t* all_reducer_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_reducer_cores;
    tt_l1_ptr uint32_t* all_reducer_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_reducer_cores;

    uint32_t reduce_core_index = (cur_batch * num_cores_per_batch) / num_cores_per_head + cur_head_group;
    uint32_t reduce_core_noc_x = all_reducer_noc_x[reduce_core_index];
    uint32_t reduce_core_noc_y = all_reducer_noc_y[reduce_core_index];

    const uint64_t in0_sender_semaphore_noc_addr =
        get_noc_addr(reduce_core_noc_x, reduce_core_noc_y, reducer_semaphore_addr);

    constexpr uint32_t out_chunk_tiles = PNHt * vDHt;
    uint32_t num_cores_to_wait = num_cores_per_head - 1;
    if (num_cores_per_head > k_num_chunks) {
        num_cores_to_wait = k_num_chunks - 1;
    }

    constexpr uint32_t cb_intermed_out =
        tt::CBIndex::c_19;  // this cb holds the output intermediates from other worker cores
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

    // generate and send scaler to compute
    // These helper functions respect tile size of CBs (ie. no need for special handling of tiny tiles)
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);
    generate_reduce_scaler(cb_zero_in, zero_scalar_packed);
    generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed);

    // =========================================================================
    // KV Cache Multicast (BRISC - page-level pipelining with NCRISC DRAM reads)
    // For sharded: NCRISC signals after each page, BRISC multicasts each page immediately
    // For interleaved: NCRISC signals once per chunk, BRISC multicasts entire chunk
    // =========================================================================
    if (is_mcast_sender) {
        constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
        constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
        uint32_t k_chunk_tiles = Sk_chunk_t_dynamic * DHt;
        uint32_t k_chunk_bytes = k_chunk_tiles * k_tile_bytes;

        // Set up semaphores
        const uint32_t mcast_semaphore_addr = get_semaphore(mcast_semaphore_id);
        volatile tt_l1_ptr uint32_t* mcast_semaphore_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_semaphore_addr);
        const uint32_t ncrisc_brisc_sync_addr = get_semaphore(ncrisc_brisc_sync_semaphore_id);
        volatile tt_l1_ptr uint32_t* ncrisc_brisc_sync_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ncrisc_brisc_sync_addr);

        // Get K buffer base address - same as reader, addresses are deterministic
        uint32_t k_write_ptr = get_write_ptr(cb_k_in);

        // Set up multicast addresses
        // Must use NOC_0 for multicast due to grid wrapping requirements
        constexpr uint8_t MCAST_NOC = 0;  // NOC_0
        const uint64_t mcast_noc_addr =
            get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, 0);
        const uint64_t mcast_sem_addr = mcast_noc_addr | mcast_semaphore_addr;

        // Set local mcast semaphore to valid (will be multicast per chunk)
        noc_semaphore_set(mcast_semaphore_ptr, 1);  // MCAST_VALID = 1

        // Get NOC address of sync semaphore for signaling back to NCRISC
        const uint64_t ncrisc_brisc_sync_noc_addr = get_noc_addr(ncrisc_brisc_sync_addr);

        // Multicast loop
        // Semaphore counting: for each chunk, NCRISC sends k_num_pages (or 1) signals, BRISC sends 1 ack
        // Chunk N starts at semaphore value N * (k_num_pages + 1) for sharded, N * 2 for interleaved
        uint32_t chunk_count = 0;
        DPRINT << "starting multicast loop" << ENDL();
        for (uint32_t cur_head = cur_head_group * num_heads_per_core;
             cur_head < cur_head_group * num_heads_per_core + num_heads_per_core;
             ++cur_head) {
            for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
                DeviceZoneScopedN("mcast-sender-multicast");

                if constexpr (kv_is_sharded) {
                    // Sharded: page-level pipelining - multicast each page as it arrives
                    // Base semaphore value for this chunk
                    uint32_t chunk_base = chunk_count * (k_num_pages + 1);
                    for (uint32_t page = 0; page < k_num_pages; ++page) {
                        // Wait for NCRISC to signal this page is ready
                        uint32_t expected = chunk_base + page + 1;
                        noc_semaphore_wait(ncrisc_brisc_sync_ptr, expected);
                        DPRINT << "done waiting for page:" << page << ENDL();

                        // Calculate page address directly - addresses increment in lockstep
                        uint32_t page_addr = k_write_ptr + page * k_page_size;

                        // Multicast this page immediately to other cores in S block
                        uint64_t mcast_dest_addr = mcast_noc_addr | page_addr;
                        noc_async_write_multicast(
                            page_addr, mcast_dest_addr, k_page_size, num_mcast_dests, false, MCAST_NOC);
                    }
                } else {
                    // Interleaved: chunk-level - wait for entire chunk, then multicast
                    uint32_t expected = chunk_count * 2 + 1;
                    noc_semaphore_wait(ncrisc_brisc_sync_ptr, expected);

                    // Multicast entire chunk from base address
                    uint64_t mcast_dest_addr = mcast_noc_addr | k_write_ptr;
                    noc_async_write_multicast(
                        k_write_ptr, mcast_dest_addr, k_chunk_bytes, num_mcast_dests, false, MCAST_NOC);
                }

                // After all data for this chunk: barrier and signal receivers
                noc_async_write_barrier();
                noc_semaphore_set_multicast(mcast_semaphore_addr, mcast_sem_addr, num_mcast_dests, false, MCAST_NOC);

                // Signal NCRISC that multicast is done - safe to reuse buffer
                noc_semaphore_inc(ncrisc_brisc_sync_noc_addr, 1);
                chunk_count++;
            }
        }
    }

#ifdef SKIP_REDUCTION_DEBUG
    // DEBUG: Skip worker/reducer to isolate DRAM streaming + multicast
    return;
#endif

    if (is_worker) {
        DeviceZoneScopedN("writer-worker");
        ASSERT(num_heads_per_core == 1);  // if there are workers, then head must be split across workers so there
                                          // should not be more than one head per core
        worker_compute<out_chunk_tiles, cb_out_worker, cb_out_m, cb_out_l, cb_intermed_out, PNHt>(
            in0_sender_semaphore_noc_addr, worker_id_for_reduce, reduce_core_noc_x, reduce_core_noc_y);
        noc_async_atomic_barrier();
        return;
    }

    // *** Reducer Compute Below ***
    constexpr uint32_t tile_bytes_intermed = get_tile_size(cb_intermed_out);

    uint64_t intermed_l1_read_addr = get_noc_addr(get_read_ptr(cb_intermed_out));

    volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reducer_semaphore_addr);

    // Generate causal mask (MLA decode is always causal)
    // Mask tiles use q_tile_height for proper tiny tile support
    generate_mask<cb_mask_in, PNHt, q_tile_height>(k_num_chunks, Sk_chunk_t_dynamic, cur_pos);

    noc_async_write_barrier();  // #19201 BH hang workaround

    for (uint32_t cur_head = cur_head_group * num_heads_per_core;
         cur_head < cur_head_group * num_heads_per_core + num_heads_per_core;
         ++cur_head) {
        DeviceZoneScopedN("writer-reducer-loop");
        if (k_chunk_end - k_chunk_start < k_num_chunks) {
            ASSERT(num_heads_per_core == 1);  // if there are workers, then head must be split across workers so there
                                              // should not be more than one head per core
            // This indicates that there are computes done by other workers. Needs to wait for them and send to
            // reducer's compute Wait for compute to deliver output chunk, and write to compute again for reduction data
            // in cb_intermed_out is arranged as [o,m,l,o,m,l,...] with size (out_chunk_tiles +
            // 2*PNHt)*num_cores_to_wait wait on in0 semaphore value to become VALID (set by sender)
            noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, num_cores_to_wait);
            // noc_semaphore_set(in0_receiver_semaphore_addr_ptr, 0);

            // cb_wait_front(cb_intermed_out, num_tiles_to_wait);
            constexpr uint32_t q_read_size = out_chunk_tiles * tile_bytes_intermed;
            constexpr uint32_t ml_read_size = PNHt * tile_bytes_intermed;
            for (uint32_t block = 0; block < num_cores_to_wait; ++block) {
                cb_reserve_back(cb_out_o, out_chunk_tiles);
                cb_reserve_back(cb_m_in, PNHt);
                cb_reserve_back(cb_l_in, PNHt);

                uint32_t m_write_ptr = get_read_ptr(cb_m_in);
                noc_async_read(intermed_l1_read_addr, m_write_ptr, ml_read_size);
                intermed_l1_read_addr += ml_read_size;
                noc_async_read_barrier();
                cb_push_back(cb_m_in, PNHt);

                uint32_t l_write_ptr = get_read_ptr(cb_l_in);
                noc_async_read(intermed_l1_read_addr, l_write_ptr, ml_read_size);
                intermed_l1_read_addr += ml_read_size;
                noc_async_read_barrier();
                cb_push_back(cb_l_in, PNHt);

                uint32_t q_write_ptr = get_read_ptr(cb_out_o);
                noc_async_read(intermed_l1_read_addr, q_write_ptr, q_read_size);
                intermed_l1_read_addr += q_read_size;
                noc_async_read_barrier();
                cb_push_back(cb_out_o, out_chunk_tiles);
            }
        }
        // Output is always sharded, num_kv_heads == 1 for MLA
        // Output is already in the sharded CB, nothing to do
        noc_async_writes_flushed();
    }
}
