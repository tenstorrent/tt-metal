// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "matmul_dataflow_common.hpp"

void kernel_main() {
    // M_tiles, padded_M_tiles, M_blocks_per_core, K_tiles, padded_K_tiles come from runtime args.
    constexpr uint32_t N_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t padded_N_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t in0_tile_size = get_compile_time_arg_val(6);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(7);
    const uint32_t in0_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(8));
    const uint32_t in0_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(9));
    const uint32_t in0_valid_semaphore_addr = get_semaphore(get_compile_time_arg_val(10));
    constexpr uint32_t is_output_writer = get_compile_time_arg_val(11);
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(12);
    constexpr bool transpose_a = static_cast<bool>(get_compile_time_arg_val(13));
    constexpr bool use_offset = static_cast<bool>(get_compile_time_arg_val(14));
    constexpr bool use_out_offset = static_cast<bool>(get_compile_time_arg_val(15));

    // Load input/output addresses and range parameters
    uint32_t argidx = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_sink_core = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_dest_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_dest_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    // OFFSET_M_AXIS + OFFSET_IN0_ROW override these from on-device offsets (and recomputes per-core M).
    uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);
    // Host passes core.y here; we compute defer_write_k_block from runtime K_num_blocks below.
    const uint32_t core_y_index = get_arg_val<uint32_t>(argidx++);

    const uint32_t out_addr_rt_arg_idx = argidx;  // Output address comes next.

    // Tensor accessors (scalar CTAs 0..15; tensor accessors start at 16: in0, then output).
    constexpr auto in0_args = TensorAccessorArgs<16>();
    const auto in0_reader = TensorAccessor(in0_args, in0_addr, in0_tile_size);

    constexpr auto out_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto out_accessor = TensorAccessor(out_args, get_arg_val<uint32_t>(out_addr_rt_arg_idx), out_tile_size);

    // Variable-M: read actual M values from runtime args (after output address).
    // OFFSET_M_AXIS overrides M_tiles and M_blocks_per_core from on-device offsets.
    uint32_t M_tiles = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 1);
    const uint32_t padded_M_tiles = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 2);
    uint32_t M_blocks_per_core = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 3);
    // Read-at-offset support — only read the runtime args when the compile-time flag is set
    // (avoids any potential register pressure / dead-code propagation issues on the no-offset
    // hot path used by all backward calls). Row and K offsets are EP-only: they're 0 on the
    // host-scalar path and derived from offsets[start] on the EP path below.
    uint32_t in0_row_offset_tiles = 0U;
    uint32_t out_row_offset_tiles = 0U;
    uint32_t in0_k_offset_tiles = 0U;
    uint32_t parent_M_tiles_stride = 0U;
    uint32_t parent_K_tiles_stride = 0U;
    if constexpr (use_offset) {
        parent_M_tiles_stride = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 4);
        parent_K_tiles_stride = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 5);
    }
    // OFFSET_IN0_K / OFFSET_IN1_K overrides K_tiles from on-device offsets[start..start+2].
    uint32_t K_tiles = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 6);

#ifdef OFFSETS_ACTIVE
    // EP path: read on-device offsets and override the matching host-derived values. Each flag
    // is independent; they compose freely.
    //   OFFSET_M_AXIS:   offsets[start..start+2] → M_tiles + per-core M; publishes M on cb_ctrl.
    //   OFFSET_IN0_ROW:  also sets in0_row_offset_tiles from offsets[start].
    //   OFFSET_OUT_ROW:  also sets out_row_offset_tiles from offsets[start] when this kernel
    //                    is the writer (non-transpose_core_grid).
    //   OFFSET_IN0_K:    in0 K-slice — sets in0_k_offset_tiles + K_tiles; publishes K on cb_ctrl.
    //   OFFSET_IN1_K:    in1 K-slice — only sets K_tiles locally here; dm_in1 owns the offset
    //                    and (when OFFSET_IN0_K is not set) the cb_ctrl publish.
    {
        const uint32_t offsets_addr = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 7);
        const uint32_t offsets_start_index = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 8);
        constexpr auto offsets_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
        const auto offsets_acc = TensorAccessor(offsets_args, offsets_addr);
        // Use cb_id_in0's L1 backing as scratch — the CB is unused at kernel startup, and the
        // real in0 tile reads only begin inside the K-loop below.
        // Limitation: this reads exactly ONE page (kPageBytes) of the offsets tensor. The
        // (start, end) pair must therefore fall within page 0 — i.e. (E + 1) * sizeof(uint32_t)
        // <= kPageBytes, where E is num_experts. On Blackhole kPageBytes is typically 4 KB
        // (~1024 experts max). Larger E would need a strided / page-aware read.
        constexpr uint32_t kPageBytes = decltype(offsets_args)::AlignedPageSize;
        const uint32_t offsets_l1_addr = get_write_ptr(tt::CBIndex::c_0);
        noc_async_read(get_noc_addr(0, offsets_acc), offsets_l1_addr, kPageBytes);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* offsets_stage = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(offsets_l1_addr);
        const uint32_t row_start = offsets_stage[offsets_start_index];
        const uint32_t row_end = offsets_stage[offsets_start_index + 1U];
#ifdef OFFSET_M_AXIS
        const uint32_t in0_idx = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 9);
        const uint32_t actual_eff_M = (row_end - row_start) / 32U;
        // Empty-expert (actual=0) → M_blocks_per_core=0 (loop skipped). Still clamp M_tiles
        // to >=1 for in0_shape construction (TensorShape2D asserts d0>0); the shape isn't
        // read once the M-loop is skipped.
        M_tiles = actual_eff_M > 0U ? actual_eff_M : 1U;
#ifdef OFFSET_IN0_ROW
        in0_row_offset_tiles = row_start / 32U;
#endif
#ifdef OFFSET_OUT_ROW
        // OFFSET_OUT_ROW + this kernel is the writer (non-transpose_core_grid) → also override
        // out_row_offset_tiles. is_output_writer is a CTA constant for this kernel.
        if constexpr (is_output_writer) {
            out_row_offset_tiles = row_start / 32U;
        }
#endif
        // Per-core M split — mirrors host actual_M_tiles_per_core formula. M_blocks_per_core
        // is UNIFORM across cores (avoids breaking the sender/receiver semaphore chain when
        // some cores have less M-work). Read/write bounds checks (m_tile >= shape.logical_d0)
        // clip out-of-range tiles for the tail cores.
        constexpr uint32_t kAxisCores = IN0_AXIS_CORES;
        const uint32_t per_core = (actual_eff_M + kAxisCores - 1U) / kAxisCores;
        M_start_tile = per_core * in0_idx;
        M_end_tile = per_core * (in0_idx + 1U);
        M_blocks_per_core = (per_core + M_block_tiles - 1U) / M_block_tiles;
        // Publish (M_start, M_end, M_blocks_per_core) to compute via cb_ctrl.
        cb_reserve_back(tt::CBIndex::c_8, 1U);
        volatile tt_l1_ptr uint32_t* ctrl_l1 =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(tt::CBIndex::c_8));
        ctrl_l1[0] = M_start_tile;
        ctrl_l1[1] = M_end_tile;
        ctrl_l1[2] = M_blocks_per_core;
        cb_push_back(tt::CBIndex::c_8, 1U);
#endif  // OFFSET_M_AXIS
#ifdef OFFSET_IN0_K
        in0_k_offset_tiles = row_start / 32U;
        K_tiles = (row_end - row_start) / 32U;
        cb_reserve_back(tt::CBIndex::c_8, 1U);
        volatile tt_l1_ptr uint32_t* ctrl_l1 =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(tt::CBIndex::c_8));
        ctrl_l1[3] = K_tiles;
        cb_push_back(tt::CBIndex::c_8, 1U);
#elif defined(OFFSET_IN1_K)
        // OFFSET_IN1_K without OFFSET_IN0_K: K_tiles still needs the local override (used for
        // padded_K_tiles below) but dm_in1 owns the cb_ctrl publish for compute.
        K_tiles = (row_end - row_start) / 32U;
#endif  // OFFSET_IN0_K / OFFSET_IN1_K
    }
#endif  // OFFSETS_ACTIVE
    const uint32_t padded_K_tiles = ((K_tiles + K_block_tiles - 1U) / K_block_tiles) * K_block_tiles;

    // Storage layout: without transpose_a the input is stored as [M, K]; with it, as [K, M].
    // shape carries the MATMUL-coord effective sizes — used for bounds checks.
    const TensorShape2D in0_shape = transpose_a ? TensorShape2D(K_tiles, M_tiles, padded_K_tiles, padded_M_tiles)
                                                : TensorShape2D(M_tiles, K_tiles, padded_M_tiles, padded_K_tiles);
    const TensorShape2D out_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);

    const uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    const uint32_t defer_write_k_block = compute_defer_write_k_block(core_y_index, Y_AXIS_CORES, K_num_blocks);
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_out = tt::CBIndex::c_2;

    volatile tt_l1_ptr uint32_t* in0_valid_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_valid_semaphore_addr);
    *(in0_valid_semaphore_addr_ptr) = VALID;
    volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_receiver_semaphore_addr);

    volatile tt_l1_ptr uint32_t* in0_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_sender_semaphore_addr);
    const uint64_t in0_sender_semaphore_noc_addr =
        get_noc_addr(in0_sender_noc_x, in0_sender_noc_y, in0_sender_semaphore_addr);

    const uint64_t in0_receiver_semaphore_noc_addr =
        get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, in0_receiver_semaphore_addr);

    /**
     * This is a Row-Major output block ordering.
     * It enables reuse of the last in0 block when striding the output block N dimension.
     */

    bool k_forward = true;
    bool reuse_block = false;

    uint32_t defer_write_m_tile = 0;
    uint32_t defer_write_m_tile_end = 0;
    uint32_t defer_write_n_tile = 0;
    uint32_t defer_write_n_tile_end = 0;
    bool defer_write = false;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        const uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        const uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
        const uint32_t current_M_block_tiles = m_tile_end - m_tile;
        const uint32_t current_block_bytes = current_M_block_tiles * K_block_tiles * in0_tile_size;

        // When striding M block, in0 gets no reuse
        reuse_block = false;
        k_forward = true;
        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            const uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            const uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);

            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                if (defer_write && k_block_iter == defer_write_k_block) {
                    if constexpr (is_output_writer) {
                        do_deferred_block_write<M_block_tiles, N_block_tiles, use_out_offset>(
                            out_accessor,
                            out_shape,
                            cb_id_out,
                            out_tile_size,
                            defer_write_m_tile,
                            defer_write_m_tile_end,
                            defer_write_n_tile,
                            defer_write_n_tile_end,
                            out_row_offset_tiles);
                    }
                }

                if (reuse_block && k_block_iter == 0) {
                    // We strided an N block and this is the first k block, so we get reuse and do not need to read in0
                    reuse_block = false;
                    continue;
                }
                const uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
                cb_reserve_back(cb_id_in0, in0_block_num_tiles);

                const uint32_t in0_start_address = get_write_ptr(cb_id_in0);
                if constexpr (is_injector_core) {
                    read_in0_block_sync<M_block_tiles, K_block_tiles, transpose_a, use_offset>(
                        in0_reader,
                        in0_shape,
                        in0_start_address,
                        in0_tile_size,
                        m_tile,
                        m_tile_end,
                        k_block * K_block_tiles,
                        (k_block + 1) * K_block_tiles,
                        in0_row_offset_tiles,
                        parent_M_tiles_stride,
                        in0_k_offset_tiles,
                        parent_K_tiles_stride);
                } else {
                    // Get from previous device
                    noc_semaphore_set(in0_receiver_semaphore_addr_ptr, INVALID);
                    noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);
                    noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, VALID);
                }

                // Critical to performance for sender to push data to compute before mcasting
                // This frees sender to start next read earlier
                cb_push_back(cb_id_in0, in0_block_num_tiles);

                if (!is_sink_core) {
                    noc_semaphore_wait(in0_sender_semaphore_addr_ptr, 1);
                    noc_semaphore_set(in0_sender_semaphore_addr_ptr, 0);

                    const uint64_t in0_unicast_data_addr =
                        get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, in0_start_address);

                    /**
                     * in0 is M_block_tiles x K_block_tiles. When M block is partial, we don't need to write the
                     * padded tiles. Use `current_block_bytes`.
                     */
                    noc_async_write(in0_start_address, in0_unicast_data_addr, current_block_bytes);

#ifdef ARCH_BLACKHOLE
                    noc_async_writes_flushed();
#endif

                    noc_semaphore_set_remote(in0_valid_semaphore_addr, in0_receiver_semaphore_noc_addr);
                }
            }

            k_forward = !k_forward;
            // We get reuse on in0 when striding N block
            reuse_block = true;

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
            // When K_num_blocks == 0 (empty K-axis offset / empty expert), the deferred-write
            // trigger inside the K-loop never fires, so previously-deferred blocks would never
            // reach DRAM and the tiles keep allocator leftovers. Force every block to a direct
            // write so the kernel-side zero-init in compute lands in the output tensor.
            if (K_num_blocks == 0U) {
                defer_write = false;
            }

            if (!defer_write) {
                if constexpr (is_output_writer) {
                    do_final_block_write<M_block_tiles, N_block_tiles, use_out_offset>(
                        out_accessor,
                        out_shape,
                        cb_id_out,
                        out_tile_size,
                        m_tile,
                        m_tile_end,
                        n_tile,
                        n_tile_end,
                        out_row_offset_tiles);
                }
            }
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
