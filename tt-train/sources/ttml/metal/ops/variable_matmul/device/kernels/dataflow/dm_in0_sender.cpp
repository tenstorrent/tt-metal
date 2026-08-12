// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "matmul_dataflow_common.hpp"

void kernel_main() {
    // M_tiles, padded_M_tiles, M_blocks_per_core, K_tiles come from runtime args (padded_K_tiles
    // is derived locally below).
    constexpr uint32_t N_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t padded_N_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t in0_tile_size = get_compile_time_arg_val(6);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(7);
    Semaphore in0_sender_sem(get_compile_time_arg_val(8));
    Semaphore in0_receiver_sem(get_compile_time_arg_val(9));
    Semaphore in0_valid_sem(get_compile_time_arg_val(10));
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
    // OFFSET_ROW_MODE overrides these from on-device offsets (and recomputes per-core M).
    uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);
    // Host passes core.y here; we compute defer_write_k_block from runtime K_num_blocks below.
    const uint32_t core_y_index = get_arg_val<uint32_t>(argidx++);

    const uint32_t out_addr_rt_arg_idx = argidx;  // Output address comes next.

    // Tensor accessors (scalar CTAs 0..15; tensor accessors start at 16: in0, output, then offsets).
    constexpr auto in0_args = TensorAccessorArgs<16>();
    const auto in0_reader = TensorAccessor(in0_args, in0_addr, in0_tile_size);

    constexpr auto out_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto out_accessor = TensorAccessor(out_args, get_arg_val<uint32_t>(out_addr_rt_arg_idx), out_tile_size);

    // Variable-M: read actual M values from runtime args (after output address).
    // OFFSET_ROW_MODE overrides M_tiles and M_blocks_per_core from on-device offsets.
    uint32_t M_tiles = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 1);
    const uint32_t padded_M_tiles = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 2);
    uint32_t M_blocks_per_core = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 3);
    // Read-at-offset support — the parent-stride runtime args are read only when use_offset is set
    // (the if-constexpr keeps them out of the no-offset compile variant). Row and K offsets are
    // initialized to 0 here and overwritten below from offsets[start..start+2] when the role
    // activates them.
    uint32_t in0_row_offset_tiles = 0U;
    uint32_t out_row_offset_tiles = 0U;
    uint32_t in0_k_offset_tiles = 0U;
    uint32_t parent_M_tiles_stride = 0U;
    uint32_t parent_K_tiles_stride = 0U;
    if constexpr (use_offset) {
        parent_M_tiles_stride = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 4);
        parent_K_tiles_stride = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 5);
    }
    // OFFSET_K_MODE overrides K_tiles from on-device offsets[start..start+2].
    uint32_t K_tiles = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 6);

    Noc noc;

    // Read on-device offsets and override the matching host-derived values. One mode per role:
    //   OFFSET_ROW_MODE (InputAndOutputRow): offsets[start..start+2] → M_tiles + per-core M
    //                    (published on cb_ctrl), in0_row_offset_tiles, and — on the writer
    //                    kernel (non-transpose_core_grid) — out_row_offset_tiles.
    //   OFFSET_K_MODE   (InputAndWeightK):    in0 K-slice — sets in0_k_offset_tiles + K_tiles
    //                    and publishes K on cb_ctrl (dm_in1 reads the same offsets for its slice).
    {
        const uint32_t offsets_addr = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 7);
        const uint32_t offsets_start_index = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 8);
        constexpr auto offsets_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
        const auto offsets_acc = TensorAccessor(offsets_args, offsets_addr);
        // Use cb_id_in0's L1 backing as scratch — the CB is unused at kernel startup, and the
        // real in0 tile reads only begin inside the K-loop below.
        // Limitation: this reads exactly ONE page (page 0) of the offsets tensor. The
        // (start, end) pair must therefore fall within page 0 — i.e. (E + 1) * sizeof(uint32_t)
        // <= the offsets tensor's page size, where E is num_experts.
        const uint32_t offsets_l1_addr = CircularBuffer(tt::CBIndex::c_0).get_write_ptr();
        noc.async_read(
            offsets_acc,
            CoreLocalMem<uint32_t>(offsets_l1_addr),
            offsets_acc.get_aligned_page_size(),
            {.page_id = 0},
            {});
        noc.async_read_barrier();
        volatile tt_l1_ptr uint32_t* offsets_stage = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(offsets_l1_addr);
        const uint32_t row_start = offsets_stage[offsets_start_index];
        const uint32_t row_end = offsets_stage[offsets_start_index + 1U];
        // Contract: offsets must be tile-aligned — the windows below are addressed at tile
        // granularity (offset / 32), so a non-multiple-of-32 offset would alias the wrong tile.
        ASSERT(row_start % 32U == 0U && row_end % 32U == 0U);
#ifdef OFFSET_ROW_MODE
        const uint32_t in0_idx = get_arg_val<uint32_t>(out_addr_rt_arg_idx + 9);
        const uint32_t actual_eff_M = (row_end - row_start) / 32U;
        // Empty-expert (actual=0) → M_blocks_per_core=0 (loop skipped). Still clamp M_tiles
        // to >=1 for in0_shape construction (TensorShape2D asserts d0>0); the shape isn't
        // read once the M-loop is skipped.
        M_tiles = actual_eff_M > 0U ? actual_eff_M : 1U;
        in0_row_offset_tiles = row_start / 32U;
        // On the writer kernel (non-transpose_core_grid) also override the output write row.
        // is_output_writer is a CTA constant for this kernel.
        if constexpr (is_output_writer) {
            out_row_offset_tiles = row_start / 32U;
        }
        // Per-core M split — mirrors host M_tiles_per_core formula. M_blocks_per_core
        // is UNIFORM across cores (avoids breaking the sender/receiver semaphore chain when
        // some cores have less M-work). Read/write bounds checks (m_tile >= shape.logical_d0)
        // clip out-of-range tiles for the tail cores.
        constexpr uint32_t kAxisCores = IN0_AXIS_CORES;
        const uint32_t per_core = (actual_eff_M + kAxisCores - 1U) / kAxisCores;
        M_start_tile = per_core * in0_idx;
        M_end_tile = per_core * (in0_idx + 1U);
        M_blocks_per_core = (per_core + M_block_tiles - 1U) / M_block_tiles;
        // Publish (M_start, M_end, M_blocks_per_core) to compute via cb_ctrl.
        CircularBuffer cb_ctrl(tt::CBIndex::c_8);
        cb_ctrl.reserve_back(1U);
        volatile tt_l1_ptr uint32_t* ctrl_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_ctrl.get_write_ptr());
        ctrl_l1[0] = M_start_tile;
        ctrl_l1[1] = M_end_tile;
        ctrl_l1[2] = M_blocks_per_core;
        cb_ctrl.push_back(1U);
#endif  // OFFSET_ROW_MODE
#ifdef OFFSET_K_MODE
        in0_k_offset_tiles = row_start / 32U;
        K_tiles = (row_end - row_start) / 32U;
        CircularBuffer cb_ctrl(tt::CBIndex::c_8);
        cb_ctrl.reserve_back(1U);
        volatile tt_l1_ptr uint32_t* ctrl_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_ctrl.get_write_ptr());
        ctrl_l1[3] = K_tiles;
        cb_ctrl.push_back(1U);
#endif  // OFFSET_K_MODE
    }
    const uint32_t padded_K_tiles = ((K_tiles + K_block_tiles - 1U) / K_block_tiles) * K_block_tiles;

    // Empty expert (K_tiles==0): K_num_blocks is 0 so the K-loop is skipped and in0_shape is
    // never read; clamp its K extent to 1 only to satisfy TensorShape2D's d>0 assert.
    const uint32_t K_tiles_shape = K_tiles > 0U ? K_tiles : 1U;
    const uint32_t padded_K_tiles_shape = padded_K_tiles > 0U ? padded_K_tiles : K_block_tiles;

    // Storage layout: without transpose_a the input is stored as [M, K]; with it, as [K, M].
    // shape carries the MATMUL-coord effective sizes — used for bounds checks.
    const TensorShape2D in0_shape = transpose_a
                                        ? TensorShape2D(K_tiles_shape, M_tiles, padded_K_tiles_shape, padded_M_tiles)
                                        : TensorShape2D(M_tiles, K_tiles_shape, padded_M_tiles, padded_K_tiles_shape);
    const TensorShape2D out_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);

    const uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    const uint32_t defer_write_k_block = compute_defer_write_k_block(core_y_index, Y_AXIS_CORES, K_num_blocks);
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_out = tt::CBIndex::c_2;
    CircularBuffer cb_in0(cb_id_in0);

    in0_valid_sem.set(VALID);

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
                cb_in0.reserve_back(in0_block_num_tiles);

                const uint32_t in0_start_address = cb_in0.get_write_ptr();
                if constexpr (is_injector_core) {
                    read_in0_block_sync<M_block_tiles, K_block_tiles, transpose_a, use_offset>(
                        in0_reader,
                        in0_shape,
                        in0_start_address,
                        cb_id_in0,
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
                    // Non-injector core: receive the block from the upstream sender core.
                    in0_receiver_sem.set(INVALID);
                    in0_sender_sem.up(noc, in0_sender_noc_x, in0_sender_noc_y, 1);
                    in0_receiver_sem.wait(VALID);
                }

                // Critical to performance for the sender to push data to compute before forwarding.
                // This frees the sender to start the next read earlier.
                cb_in0.push_back(in0_block_num_tiles);

                if (!is_sink_core) {
                    in0_sender_sem.wait(1);
                    in0_sender_sem.set(0);

                    /**
                     * in0 is M_block_tiles x K_block_tiles. When M block is partial, we don't need to write the
                     * padded tiles. Use `current_block_bytes`.
                     */
                    noc.async_write(
                        CoreLocalMem<uint32_t>(in0_start_address),
                        UnicastEndpoint{},
                        current_block_bytes,
                        {},
                        {.noc_x = in0_dest_noc_x, .noc_y = in0_dest_noc_y, .addr = in0_start_address});

#ifdef ARCH_BLACKHOLE
                    noc.async_writes_flushed();
#endif

                    in0_valid_sem.relay_unicast(noc, in0_receiver_sem, in0_dest_noc_x, in0_dest_noc_y);
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
            // K_num_blocks == 0 (empty expert): the in-K-loop deferred-write trigger never
            // fires, so force direct writes to flush compute's zero-init to DRAM.
            if (K_num_blocks == 0U) {
                defer_write = false;
            }

            if (!defer_write) {
                if constexpr (is_output_writer) {
                    write_block_sync_granular<M_block_tiles, N_block_tiles, use_out_offset>(
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
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
