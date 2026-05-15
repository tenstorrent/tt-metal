// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "matmul_dataflow_common.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/fused_receiver_utils.hpp"

void kernel_main() {
    // Indices 0, 1, 9 are unused placeholders (kept for compile-time arg layout compatibility).
    // Actual M_tiles, padded_M_tiles, M_blocks_per_core come from runtime args.
    // Indices 2, 3 are unused placeholders. K_tiles comes from runtime args (variable-K).
    constexpr uint32_t N_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t padded_N_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(10);
    constexpr uint32_t in1_tile_size = get_compile_time_arg_val(11);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(12);
    constexpr uint32_t in2_tile_size = get_compile_time_arg_val(13);
    uint32_t in1_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(14));
    uint32_t in1_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(15));
    uint32_t in1_valid_semaphore_addr = get_semaphore(get_compile_time_arg_val(16));
    constexpr uint32_t is_output_writer = get_compile_time_arg_val(17);
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(18);
    constexpr uint32_t N_chunks = get_compile_time_arg_val(19);
    constexpr uint32_t N_tiles_per_chunk = get_compile_time_arg_val(20);
    constexpr bool transpose_b = static_cast<bool>(get_compile_time_arg_val(21));
    constexpr bool use_offset_in1 = static_cast<bool>(get_compile_time_arg_val(22));
    constexpr bool use_out_offset = static_cast<bool>(get_compile_time_arg_val(23));

    // Load input/output addresses and range parameters
    uint32_t argidx = 0;
    const uint32_t in1_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t in2_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_sink_core = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_dest_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_dest_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    // OFFSETS_ROLE=InputRow overrides these from on-device offsets (per-core M re-derived).
    uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);
    // Host passes core.y here; we compute defer_write_k_block from runtime K_num_blocks below.
    const uint32_t core_y_index = get_arg_val<uint32_t>(argidx++);

#ifdef FUSE_TERNARY
    // Fuse addcmul - read runtime addresses before setting out_addr_rt_arg_idx
    const uint32_t ternary_a_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t ternary_b_addr = get_arg_val<uint32_t>(argidx++);
#endif  // FUSE_TERNARY

    const uint32_t out_addr_rt_arg_idx = argidx;  // Output addresses start here (after ternary if present)

    // Tensor accessor for input tensor (CTAs 0..23 are scalar; accessor starts at 24)
    constexpr auto in1_args = TensorAccessorArgs<24>();
    const auto in1_reader = TensorAccessor(in1_args, in1_addr, in1_tile_size);

    // Always create tuple of output accessors (size = N_chunks)
    constexpr uint32_t out_tensor_args_cta_offset = in1_args.next_compile_time_args_offset();
    constexpr auto outputs_args = make_tensor_accessor_args_tuple<N_chunks, out_tensor_args_cta_offset>();
    auto outputs_tuple = make_tensor_accessor_tuple_uniform_page_size(outputs_args, out_addr_rt_arg_idx, out_tile_size);

#ifdef FUSE_BIAS
    constexpr uint32_t in2_args_cta_offset =
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<N_chunks, out_tensor_args_cta_offset>();
    constexpr auto in2_args = TensorAccessorArgs<in2_args_cta_offset>();
    const auto in2_reader = TensorAccessor(in2_args, in2_addr, in2_tile_size);
#endif

#ifdef FUSE_TERNARY
// Calculate offset for ternary_a_args
#ifdef FUSE_BIAS
    constexpr uint32_t ternary_a_args_cta_offset = in2_args.next_compile_time_args_offset();
#else
    constexpr uint32_t ternary_a_args_cta_offset =
        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<N_chunks, out_tensor_args_cta_offset>();
#endif
    constexpr uint32_t cb_id_ternary_a = tt::CBIndex::c_5;
    constexpr uint32_t cb_id_ternary_b = tt::CBIndex::c_6;

    constexpr uint32_t ternary_a_tile_size = get_tile_size(cb_id_ternary_a);
    constexpr uint32_t ternary_b_tile_size = get_tile_size(cb_id_ternary_b);

    constexpr auto ternary_a_args = TensorAccessorArgs<ternary_a_args_cta_offset>();
    constexpr auto ternary_b_args = TensorAccessorArgs<ternary_a_args.next_compile_time_args_offset()>();
    const auto ternary_a_reader = TensorAccessor(ternary_a_args, ternary_a_addr, ternary_a_tile_size);
    const auto ternary_b_reader = TensorAccessor(ternary_b_args, ternary_b_addr, ternary_b_tile_size);

#endif  // FUSE_TERNARY

    // Variable-M: read actual M values from runtime args (after output addresses)
    // OFFSETS_ROLE=InputRow overrides M_tiles and M_blocks_per_core from on-device offsets.
    uint32_t M_tiles = get_arg_val<uint32_t>(out_addr_rt_arg_idx + N_chunks);
    const uint32_t padded_M_tiles = get_arg_val<uint32_t>(out_addr_rt_arg_idx + N_chunks + 1);
    uint32_t M_blocks_per_core = get_arg_val<uint32_t>(out_addr_rt_arg_idx + N_chunks + 2);
    uint32_t in1_k_offset_tiles = 0U;
    uint32_t parent_K_tiles_stride_in1 = 0U;
    if constexpr (use_offset_in1) {
        in1_k_offset_tiles = get_arg_val<uint32_t>(out_addr_rt_arg_idx + N_chunks + 3);
        parent_K_tiles_stride_in1 = get_arg_val<uint32_t>(out_addr_rt_arg_idx + N_chunks + 4);
    }
    uint32_t out_row_offset_tiles = 0U;
    if constexpr (use_out_offset) {
        out_row_offset_tiles = get_arg_val<uint32_t>(out_addr_rt_arg_idx + N_chunks + 5);
    }
    // Variable-K: matmul-K extent from runtime; padded_K and K_num_blocks derived using
    // K_block_tiles (CTA). One cached program services any K value.
    // OFFSETS_ROLE=InputK/WeightK overrides K_tiles from on-device offsets[start..start+2].
    uint32_t K_tiles = get_arg_val<uint32_t>(out_addr_rt_arg_idx + N_chunks + 6);

#ifdef OFFSETS_ROLE
    // EP path: read offsets from a 1-D UINT32 ROW_MAJOR device tensor.
    //   OutputRow (1): offsets[start] -> out_row_offset_tiles (write-at-offset row).
    //   InputRow  (2): offsets[start..start+2] -> M_tiles + per-core M_start/M_end/
    //                  M_blocks_per_core. dm_in0_sender publishes the per-core M values
    //                  via cb_ctrl; this kernel re-derives them locally.
    //   InputK    (3): offsets[start..start+2] -> K_tiles only (in0 side owns the offset);
    //                  no cb_ctrl write (dm_in0_sender publishes for compute).
    //   WeightK   (4): offsets[start..start+2] -> in1_k_offset_tiles + K_tiles. Publishes
    //                  K_tiles on cb_ctrl[3].
    {
        constexpr uint32_t kRole = OFFSETS_ROLE;
        static_assert(
            kRole == 1U || kRole == 2U || kRole == 3U || kRole == 4U || kRole == 5U || kRole == 6U,
            "Unsupported OFFSETS_ROLE value.");
        const uint32_t offsets_addr = get_arg_val<uint32_t>(out_addr_rt_arg_idx + N_chunks + 7);
        const uint32_t offsets_start_index = get_arg_val<uint32_t>(out_addr_rt_arg_idx + N_chunks + 8);
        constexpr uint32_t offsets_args_cta_offset =
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<N_chunks, out_tensor_args_cta_offset>();
        constexpr auto offsets_args = TensorAccessorArgs<offsets_args_cta_offset>();
        const auto offsets_acc = TensorAccessor(offsets_args, offsets_addr);

        // Use the in1 CB's L1 write pointer as scratch — unused at kernel startup; the
        // real in1 tile reads only begin inside the K-loop below.
        constexpr uint32_t kPageBytes = decltype(offsets_args)::AlignedPageSize;
        uint32_t offsets_l1_addr = get_write_ptr(tt::CBIndex::c_1);
        noc_async_read(get_noc_addr(0, offsets_acc), offsets_l1_addr, kPageBytes);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* offsets_stage = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(offsets_l1_addr);
#if OFFSETS_ROLE == 1 || OFFSETS_ROLE == 2 || OFFSETS_ROLE == 5
        {
            const uint32_t in0_idx = get_arg_val<uint32_t>(out_addr_rt_arg_idx + N_chunks + 9);
            const uint32_t row_start = offsets_stage[offsets_start_index];
            const uint32_t row_end = offsets_stage[offsets_start_index + 1U];
            const uint32_t actual_eff_M = (row_end - row_start) / 32U;
            // Empty-expert (actual=0) → M_blocks_per_core=0 (loop skipped). Still clamp
            // M_tiles to >=1 for shape construction (TensorShape2D asserts d0>0).
            M_tiles = actual_eff_M > 0U ? actual_eff_M : 1U;
#if OFFSETS_ROLE == 1 || OFFSETS_ROLE == 5
            // OutputRow (1) / InputAndOutputRow (5) + this kernel is the writer
            // (transpose_core_grid) → override out_row_offset_tiles. dm_in0_sender publishes
            // M values to cb_ctrl; we re-derive them locally here (both kernels read the
            // same offsets).
            if constexpr (is_output_writer) {
                out_row_offset_tiles = row_start / 32U;
            }
#endif
            constexpr uint32_t kAxisCores = IN0_AXIS_CORES;
            const uint32_t per_core = (actual_eff_M + kAxisCores - 1U) / kAxisCores;
            // Uniform M_blocks_per_core across cores — matches dm_in0_sender; bounds checks
            // clip out-of-range reads/writes for the tail cores.
            M_start_tile = per_core * in0_idx;
            M_end_tile = per_core * (in0_idx + 1U);
            M_blocks_per_core = (per_core + M_block_tiles - 1U) / M_block_tiles;
        }
#elif OFFSETS_ROLE == 3
        {
            const uint32_t row_start = offsets_stage[offsets_start_index];
            const uint32_t row_end = offsets_stage[offsets_start_index + 1U];
            K_tiles = (row_end - row_start) / 32U;
        }
#elif OFFSETS_ROLE == 4
        {
            const uint32_t row_start = offsets_stage[offsets_start_index];
            const uint32_t row_end = offsets_stage[offsets_start_index + 1U];
            in1_k_offset_tiles = row_start / 32U;
            K_tiles = (row_end - row_start) / 32U;
            // Publish K_tiles to compute via cb_ctrl[3].
            cb_reserve_back(tt::CBIndex::c_8, 1U);
            volatile tt_l1_ptr uint32_t* ctrl_l1 =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(tt::CBIndex::c_8));
            ctrl_l1[3] = K_tiles;
            cb_push_back(tt::CBIndex::c_8, 1U);
        }
#elif OFFSETS_ROLE == 6
        {
            // InputAndWeightK: this kernel owns in1_k_offset + K_tiles override. dm_in0_sender
            // owns in0_k_offset + the cb_ctrl publish of K_tiles for compute.
            const uint32_t row_start = offsets_stage[offsets_start_index];
            const uint32_t row_end = offsets_stage[offsets_start_index + 1U];
            in1_k_offset_tiles = row_start / 32U;
            K_tiles = (row_end - row_start) / 32U;
        }
#endif
    }
#endif  // OFFSETS_ROLE
    const uint32_t padded_K_tiles = ((K_tiles + K_block_tiles - 1U) / K_block_tiles) * K_block_tiles;

    // Storage layout: without transpose_b the weight is stored as [K, N]; with it, as [N, K].
    const TensorShape2D in1_shape = transpose_b ? TensorShape2D(N_tiles, K_tiles, padded_N_tiles, padded_K_tiles)
                                                : TensorShape2D(K_tiles, N_tiles, padded_K_tiles, padded_N_tiles);
    const TensorShape2D out_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);
    const TensorShape2D out0_shape(M_tiles, N_tiles_per_chunk, padded_M_tiles, N_tiles_per_chunk);

    const uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    // Compute defer_write_k_block from runtime K_num_blocks. Stagger across Y_AXIS_CORES so
    // deferred output writes from different cores spread across the next block's K loop
    // (latency hiding). Clamp to the last K iter — without this, K-axis OffsetsRoles that
    // shrink K at runtime can leave the check never firing, deadlocking cb_id_out (cap 2)
    // once M_blocks_per_core * N_blocks_per_core - 1 >= 3.
    constexpr uint32_t kYAxisCores = Y_AXIS_CORES;
    const uint32_t k_blocks_per_axis_core = (K_num_blocks + kYAxisCores - 1U) / kYAxisCores;
    uint32_t defer_write_k_block = core_y_index * k_blocks_per_axis_core;
    if (K_num_blocks > 0U) {
        defer_write_k_block = std::min(defer_write_k_block, K_num_blocks - 1U);
    }
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_out = tt::CBIndex::c_2;
#ifdef FUSE_BIAS
    constexpr uint32_t cb_id_in2 = tt::CBIndex::c_4;
#endif

#ifdef FUSE_AG
    // Receiver for ccl fusing
    MinimalMatmulOpReceiver fused_op_receiver;
    uint32_t fused_op_rt_args_idx = out_addr_rt_arg_idx + N_chunks;
    uint32_t num_devices = get_arg_val<uint32_t>(fused_op_rt_args_idx);
    uint32_t num_k_blocks = get_arg_val<uint32_t>(fused_op_rt_args_idx + 1);
    uint8_t k_block_device_expected[num_k_blocks]{};
    uint8_t k_block_device_received[num_k_blocks]{};
    uint32_t device_k_block_counts[num_devices]{};
    uint32_t device_k_block_start_ids[num_devices]{};
    uint32_t forward_k_block_schedule[num_k_blocks]{};
    if constexpr (is_injector_core) {
        fused_op_receiver = MinimalMatmulOpReceiver(
            false,
            fused_op_rt_args_idx,
            k_block_device_expected,
            k_block_device_received,
            device_k_block_counts,
            device_k_block_start_ids,
            forward_k_block_schedule);
    }
#endif

    volatile tt_l1_ptr uint32_t* in1_valid_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_valid_semaphore_addr);
    *(in1_valid_semaphore_addr_ptr) = VALID;
    volatile tt_l1_ptr uint32_t* in1_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* in1_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_sender_semaphore_addr);
    const uint64_t in1_sender_semaphore_noc_addr =
        get_noc_addr(in1_sender_noc_x, in1_sender_noc_y, in1_sender_semaphore_addr);

    const uint64_t in1_receiver_semaphore_noc_addr =
        get_noc_addr(in1_dest_noc_x, in1_dest_noc_y, in1_receiver_semaphore_addr);

    const uint64_t in1_unicast_data_base_addr = get_noc_addr(in1_dest_noc_x, in1_dest_noc_y, 0);

    constexpr uint32_t full_N_tiles_bytes = N_block_tiles * in1_tile_size;

    bool k_forward = true;

    uint32_t defer_write_m_tile = 0;
    uint32_t defer_write_m_tile_end = 0;
    uint32_t defer_write_n_tile = 0;
    uint32_t defer_write_n_tile_end = 0;
    bool defer_write = false;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
#ifdef FUSE_AG
        if constexpr (is_injector_core) {
            fused_op_receiver.reset();
        }
#endif

        k_forward = true;

        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            uint32_t current_N_block_tiles = n_tile_end - n_tile;
            uint32_t current_N_tiles_bytes = current_N_block_tiles * in1_tile_size;
            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                if (defer_write && k_block_iter == defer_write_k_block) {
                    if constexpr (is_output_writer) {
                        cb_wait_front(cb_id_out, out_block_num_tiles);
                        uint32_t out_read_ptr = get_read_ptr(cb_id_out);

                        // write_block_sync_split is more generic (support multiple output tensors)
                        // But for N_chunks == 1 (non-split minimal_matmul), write_block_sync should be faster
                        if constexpr (N_chunks == 1) {
                            write_block_sync<M_block_tiles, N_block_tiles, use_out_offset>(
                                std::get<0>(outputs_tuple),
                                out_shape,
                                out_read_ptr,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile,
                                defer_write_n_tile_end,
                                out_row_offset_tiles);
                        } else {
                            write_block_sync_split<M_block_tiles, N_block_tiles, N_chunks, N_tiles_per_chunk>(
                                outputs_tuple,
                                out0_shape,
                                out_read_ptr,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile,
                                defer_write_n_tile_end);
                        }
                        cb_pop_front(cb_id_out, out_block_num_tiles);
                    }
                }

                uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
                cb_reserve_back(cb_id_in1, in1_block_num_tiles);

                uint32_t in1_start_address = get_write_ptr(cb_id_in1);
                if constexpr (is_injector_core) {
#ifdef FUSE_AG
                    if (is_injector_core) {
                        k_block =
                            fused_op_receiver.compute_actual_k_block_iter(n_block_iter == 0, k_block_iter, k_forward);
                    }
#endif
                    read_in1_block_sync<K_block_tiles, N_block_tiles, transpose_b, use_offset_in1>(
                        in1_reader,
                        in1_shape,
                        in1_start_address,
                        in1_tile_size,
                        k_block * K_block_tiles,
                        (k_block + 1) * K_block_tiles,
                        n_tile,
                        n_tile_end,
                        in1_k_offset_tiles,
                        parent_K_tiles_stride_in1);
                } else {
                    noc_semaphore_set(in1_receiver_semaphore_addr_ptr, INVALID);
                    noc_semaphore_inc(in1_sender_semaphore_noc_addr, 1);
                    noc_semaphore_wait(in1_receiver_semaphore_addr_ptr, VALID);
                }

                // Critical to performance for sender to push data to compute before mcasting
                // This frees sender to start next read earlier
                cb_push_back(cb_id_in1, in1_block_num_tiles);

                if (!is_sink_core) {
                    noc_semaphore_wait(in1_sender_semaphore_addr_ptr, 1);
                    noc_semaphore_set(in1_sender_semaphore_addr_ptr, 0);

                    /**
                     * in1 is K_block_tiles x N_block_tiles. When N block is partial, we don't need to write the
                     * padded tiles. For each tile in the K block, write only the non-padded N tiles. Use
                     * `current_N_tiles_bytes`.
                     */
                    for (uint32_t i = 0; i < K_block_tiles; i++) {
                        uint64_t in1_unicast_data_addr = in1_unicast_data_base_addr | in1_start_address;
                        noc_async_write(in1_start_address, in1_unicast_data_addr, current_N_tiles_bytes);
                        in1_start_address += full_N_tiles_bytes;
                    }

#ifdef ARCH_BLACKHOLE
                    noc_async_writes_flushed();
#endif

                    noc_semaphore_set_remote(in1_valid_semaphore_addr, in1_receiver_semaphore_noc_addr);
                }
            }
#ifdef FUSE_BIAS
            if constexpr (!is_output_writer) {
                cb_reserve_back(cb_id_in2, N_block_tiles);

                uint32_t l1_write_addr_in2 = get_write_ptr(cb_id_in2);
                for (uint32_t n_tile_id = n_tile; n_tile_id < n_tile_end; n_tile_id++) {
                    noc_async_read_tile(n_tile_id, in2_reader, l1_write_addr_in2);
                    l1_write_addr_in2 += in2_tile_size;
                }
                noc_async_read_barrier();

                cb_push_back(cb_id_in2, N_block_tiles);
            }
#endif

#ifdef FUSE_TERNARY
            if constexpr (!is_output_writer) {
                read_ternary_blocks_sync<M_block_tiles, N_block_tiles>(
                    ternary_a_reader,
                    ternary_b_reader,
                    out_shape,
                    cb_id_ternary_a,
                    cb_id_ternary_b,
                    ternary_a_tile_size,
                    ternary_b_tile_size,
                    m_tile,
                    m_tile_end,
                    n_tile,
                    n_tile_end);
            }
#endif  // FUSE_TERNARY

            k_forward = !k_forward;
            // We have an output block to write out

            defer_write_m_tile = m_tile;
            defer_write_m_tile_end = m_tile_end;
            defer_write_n_tile = n_tile;
            defer_write_n_tile_end = n_tile_end;
            /**
             * If this isn't the last output block, defer writing until the defer_k_write_block iteration
             * of the next output block.
             */
            defer_write = !((m_block_iter == M_blocks_per_core - 1) && (n_block_iter == (N_blocks_per_core - 1)));
            defer_write = defer_write && !is_injector_core;

            if (!defer_write) {
                if constexpr (is_output_writer) {
                    // write_block_sync_granular_split is more generic (support multiple output tensors)
                    // But for N_chunks == 1 (non-split minimal_matmul), write_block_sync_granular should be faster
                    if constexpr (N_chunks == 1) {
                        write_block_sync_granular<M_block_tiles, N_block_tiles, use_out_offset>(
                            std::get<0>(outputs_tuple),
                            out_shape,
                            cb_id_out,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
                            n_tile,
                            n_tile_end,
                            out_row_offset_tiles);
                    } else {
                        write_block_sync_granular_split<M_block_tiles, N_block_tiles, N_chunks, N_tiles_per_chunk>(
                            outputs_tuple,
                            out0_shape,
                            cb_id_out,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
                            n_tile,
                            n_tile_end);
                    }
                }
            }
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
