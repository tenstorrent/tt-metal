// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#endif

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/tilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// Create Q Heads micro-op
//
// Gathers data from 12x8 sender cores to 4x2 receiver cores, then tilizes.
// Each sender row maps to a different receiver core:
//   Row 0 → (0,1), Row 1 → (1,1), Row 2 → (2,1), Row 3 → (3,1)
//   Row 4 → (0,2), Row 5 → (1,2), Row 6 → (2,2), Row 7 → (3,2)
//
// Memory layout for tilization (max tilize dim = 256):
//   Phase 1: First 8 halves of QNOPE - shape [8, 256] → 8 tiles
//   Phase 2: Second 8 halves of QNOPE - shape [8, 256] → 8 tiles
//   Phase 3: QROPE - shape [8, 64] → 2 tiles
//
// Sender writes:
//   - QNOPE cores: Split 512 elements into two 256-element halves
//     - First half → tight row-major at offset = col * 256
//     - Second half → tight row-major at offset = (8*256 + col*256)
//     - Signals nope_phase1_semaphore after first half
//     - Signals nope_phase2_semaphore after second half
//   - QROPE cores: Write 2 heads × 64 elements = 128 elements
//     - Offset = (8*512) + 2*qrope_col*64
//     - Signals rope_semaphore once
//
// Semaphore allocation (reuses existing pre_sdpa semaphores):
//   - nope_phase1_semaphore_id = 2 (reuse gather_noc0_receiver_semaphore_id)
//   - nope_phase2_semaphore_id = 3 (reuse gather_noc1_receiver_semaphore_id)
//   - rope_semaphore_id = 0 (reuse mcast_data_sender_semaphore_id)
// ============================================================================
struct CreateQHeads {
    // ========================================================================
    // Runtime args structs
    // ========================================================================

    // Sender args
    struct SenderArgs {
        // Grid info
        uint32_t sender_grid_start_x;
        uint32_t sender_grid_start_y;
        // Data sizes (in bytes)
        uint32_t qnope_data_size_bytes;  // 512 elements * 2 = 1024 bytes
        uint32_t qrope_head_size_bytes;  // 64 elements * 2 = 128 bytes (per head)
        uint32_t head_stride_bytes;      // 576 elements * 2 = 1152 bytes
        uint32_t qnope_cols;             // 8
        // CB indices
        uint32_t qnope_cb;
        uint32_t qrope_cb;
        uint32_t src_num_pages;
        // Semaphores (3 separate semaphores for race-free synchronization)
        uint32_t nope_phase1_semaphore_id;  // QNOPE signals after first half
        uint32_t nope_phase2_semaphore_id;  // QNOPE signals after second half
        uint32_t rope_semaphore_id;         // QROPE signals after completion
        // Target NOC coordinates (8 rows, packed: lower 16 bits = x, upper 16 bits = y)
        uint32_t target_noc_coords[8];
        // Runtime arg - destination address
        uint32_t receiver_data_addr;
    };

    // Receiver args - multi-phase synchronization for tilization
    struct ReceiverArgs {
        // Semaphores (3 separate semaphores for race-free synchronization)
        uint32_t nope_phase1_semaphore_id;  // Wait for QNOPE first halves
        uint32_t nope_phase2_semaphore_id;  // Wait for QNOPE second halves
        uint32_t rope_semaphore_id;         // Wait for QROPE
        uint32_t num_nope_senders;          // Number of QNOPE senders (8)
        uint32_t num_rope_senders;          // Number of QROPE senders (4)
        uint32_t receiver_in_cb;            // Input CB where senders write row-major data
        uint32_t out_cb;                    // Output CB for tilized data
        uint32_t nope_tiles;                // Tiles per NOPE phase (8 for [8,256])
        uint32_t rope_tiles;                // Tiles for ROPE phase (2 for [8,64])
    };

    // Compute args for tilization
    struct ComputeArgs {
        uint32_t receiver_in_cb;  // Input CB with row-major data
        uint32_t out_cb;          // Output CB for tilized data
        uint32_t nope_tiles;      // Tiles per NOPE phase (8)
        uint32_t rope_tiles;      // Tiles for ROPE phase (2)
    };

    // ========================================================================
    // Op - the actual operation
    //
    // Template parameters:
    //   IsSenderCore: true if this core is a sender (QNOPE or QROPE)
    //   IsReceiverCore: true if this core is a receiver (SDPA input)
    //   setup_sharded_input: true to call setup_sharded_buffer (standalone op)
    //   pop_src: whether to pop the source CB after sending
    //   use_cb_output: true to use cb_reserve_back/cb_push_back
    // ========================================================================
    template <bool IsSenderCore, bool IsReceiverCore, bool setup_sharded_input, bool pop_src, bool use_cb_output>
    class Op {
    public:
        // Overload for sender args
        void operator()([[maybe_unused]] const SenderArgs& args) {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
            if constexpr (IsSenderCore) {
                sender_impl(args);
            }
#endif
        }

        // Overload for receiver args
        void operator()([[maybe_unused]] const ReceiverArgs& args) {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
            if constexpr (IsReceiverCore) {
                receiver_impl(args);
            }
#endif
        }

        // Overload for compute args - tilization
        // Only runs when use_cb_output is true (tilization enabled)
        void operator()([[maybe_unused]] const ComputeArgs& args) {
#if defined(COMPILE_FOR_TRISC)
            if constexpr (IsReceiverCore && use_cb_output) {
                compute_impl(args);
            }
#endif
        }

    private:
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
        FORCE_INLINE void sender_impl(const SenderArgs& args) {
            // Compute my row and column indices within the sender grid
            uint32_t my_col = my_logical_x_ - args.sender_grid_start_x;
            uint32_t my_row = my_logical_y_ - args.sender_grid_start_y;

            // Determine if this is a Qnope or Qrope core based on column
            bool is_qnope_core = my_col < args.qnope_cols;

            // Select the appropriate CB
            uint32_t src_cb = is_qnope_core ? args.qnope_cb : args.qrope_cb;

            // Setup sharded input buffer if needed (standalone op)
            if constexpr (setup_sharded_input) {
                unified_kernels::setup_sharded_buffer(src_cb, args.src_num_pages);
            }

            // Wait for source CB data to be ready
            cb_wait_front(src_cb, args.src_num_pages);

            // Get source address from CB
            uint32_t src_addr = get_read_ptr(src_cb);

            // Get target NOC coordinates based on row (unpacked from uint32)
            uint32_t packed_coords = args.target_noc_coords[my_row];
            uint32_t target_noc_x = packed_coords & 0xFFFF;          // Lower 16 bits
            uint32_t target_noc_y = (packed_coords >> 16) & 0xFFFF;  // Upper 16 bits

            const uint64_t dst_noc_coord = get_noc_addr(target_noc_x, target_noc_y, 0);
            uint32_t half_qnope_data_size_bytes = args.qnope_data_size_bytes / 2;

            if (is_qnope_core) {
                // QNOPE core: Split 512 elements into two 256-element halves for tilization
                // Memory layout enables tilize of [8, 256] shapes:
                //   First halves: cols 0-7 packed at offsets 0, 256, 512, ... (row-major [8, 256])
                //   Second halves: cols 0-7 packed at offsets 2048, 2304, ... (row-major [8, 256])

                uint32_t phase1_semaphore_addr = get_semaphore(args.nope_phase1_semaphore_id);
                uint32_t phase2_semaphore_addr = get_semaphore(args.nope_phase2_semaphore_id);
                uint64_t phase1_semaphore_noc_addr = dst_noc_coord | (uint64_t)phase1_semaphore_addr;
                uint64_t phase2_semaphore_noc_addr = dst_noc_coord | (uint64_t)phase2_semaphore_addr;

                // First half: tight row-major packing
                uint32_t dst_offset_0 = my_col * half_qnope_data_size_bytes;
                uint64_t dst_data_noc_addr_0 = dst_noc_coord | (uint64_t)(args.receiver_data_addr + dst_offset_0);
                noc_async_write(src_addr, dst_data_noc_addr_0, half_qnope_data_size_bytes);
                noc_async_write_barrier();
                noc_semaphore_inc(phase1_semaphore_noc_addr, 1);

                // Second half: continues after first block
                uint32_t dst_offset_1 = (args.qnope_cols * half_qnope_data_size_bytes) + dst_offset_0;
                uint64_t dst_data_noc_addr_1 = dst_noc_coord | (uint64_t)(args.receiver_data_addr + dst_offset_1);
                noc_async_write(src_addr + half_qnope_data_size_bytes, dst_data_noc_addr_1, half_qnope_data_size_bytes);
                noc_async_write_barrier();
                noc_semaphore_inc(phase2_semaphore_noc_addr, 1);
            } else {
                // QROPE core: Write 2 heads × 64 elements = 128 elements
                // Memory layout: after all QNOPE data, QROPE is packed row-major [8, 64]
                uint32_t rope_semaphore_addr = get_semaphore(args.rope_semaphore_id);
                uint64_t rope_semaphore_noc_addr = dst_noc_coord | (uint64_t)rope_semaphore_addr;
                uint32_t qrope_col = my_col - args.qnope_cols;
                // Offset starts after full QNOPE region (8 cols × 512 elements each)
                uint32_t dst_offset =
                    (args.qnope_cols * args.qnope_data_size_bytes) + (2 * qrope_col * args.qrope_head_size_bytes);
                uint64_t dst_data_noc_addr = dst_noc_coord | (uint64_t)(args.receiver_data_addr + dst_offset);
                noc_async_write(src_addr, dst_data_noc_addr, args.qrope_head_size_bytes * 2);
                noc_async_write_barrier();
                noc_semaphore_inc(rope_semaphore_noc_addr, 1);
            }

            // Pop source CB after sending
            if constexpr (pop_src) {
                cb_pop_front(src_cb, args.src_num_pages);
            }
        }

        FORCE_INLINE void receiver_impl(const ReceiverArgs& args) {
            // Multi-phase receiver for tilization with 3 separate semaphores
            // Each phase has its own semaphore to prevent race conditions
            // Senders write directly to our L1 (receiver_in_cb's address space)
            // We coordinate with compute via CB push/wait

            // Get semaphore addresses for all 3 phases
            uint32_t phase1_semaphore_addr = get_semaphore(args.nope_phase1_semaphore_id);
            uint32_t phase2_semaphore_addr = get_semaphore(args.nope_phase2_semaphore_id);
            uint32_t rope_semaphore_addr = get_semaphore(args.rope_semaphore_id);

            volatile tt_l1_ptr uint32_t* phase1_semaphore_ptr = (volatile tt_l1_ptr uint32_t*)phase1_semaphore_addr;
            volatile tt_l1_ptr uint32_t* phase2_semaphore_ptr = (volatile tt_l1_ptr uint32_t*)phase2_semaphore_addr;
            volatile tt_l1_ptr uint32_t* rope_semaphore_ptr = (volatile tt_l1_ptr uint32_t*)rope_semaphore_addr;

            if (args.num_nope_senders > 0) {
                noc_semaphore_wait(phase1_semaphore_ptr, args.num_nope_senders);
                noc_semaphore_set(phase1_semaphore_ptr, 0);
            }

            if constexpr (use_cb_output) {
                cb_reserve_back(args.receiver_in_cb, args.nope_tiles);
                cb_push_back(args.receiver_in_cb, args.nope_tiles);
            }

            if (args.num_nope_senders > 0) {
                noc_semaphore_wait(phase2_semaphore_ptr, args.num_nope_senders);
                noc_semaphore_set(phase2_semaphore_ptr, 0);
            }

            if constexpr (use_cb_output) {
                cb_reserve_back(args.receiver_in_cb, args.nope_tiles);
                cb_push_back(args.receiver_in_cb, args.nope_tiles);
            }

            if (args.num_rope_senders > 0) {
                noc_semaphore_wait(rope_semaphore_ptr, args.num_rope_senders);
                noc_semaphore_set(rope_semaphore_ptr, 0);
            }

            if constexpr (use_cb_output) {
                cb_reserve_back(args.receiver_in_cb, args.rope_tiles);
                cb_push_back(args.receiver_in_cb, args.rope_tiles);
            }
        }

#endif

#if defined(COMPILE_FOR_TRISC)
        FORCE_INLINE void compute_impl(const ComputeArgs& args) {
            // Tilize row-major data from input CB to output CB
            // Three phases with per-phase CB management:
            //   Phase 1: NOPE first halves [8, 256] → 8 tiles
            //   Phase 2: NOPE second halves [8, 256] → 8 tiles
            //   Phase 3: ROPE [8, 64] → 2 tiles
            //
            // Each phase does: wait→reserve→tilize→push→pop
            // This ensures proper hardware state between tilize_block calls.

            compute_kernel_hw_startup(args.receiver_in_cb, args.out_cb);
            tilize_init(args.receiver_in_cb, args.nope_tiles, args.out_cb);

            // Phase 1: Tilize first NOPE block [8, 256] → 8 tiles
            cb_wait_front(args.receiver_in_cb, args.nope_tiles);
            cb_reserve_back(args.out_cb, args.nope_tiles);
            tilize_block(args.receiver_in_cb, args.nope_tiles, args.out_cb);
            cb_push_back(args.out_cb, args.nope_tiles);
            cb_pop_front(args.receiver_in_cb, args.nope_tiles);

            // Phase 2: Tilize second NOPE block [8, 256] → 8 tiles
            cb_wait_front(args.receiver_in_cb, args.nope_tiles);
            cb_reserve_back(args.out_cb, args.nope_tiles);
            tilize_block(args.receiver_in_cb, args.nope_tiles, args.out_cb);
            cb_push_back(args.out_cb, args.nope_tiles);
            cb_pop_front(args.receiver_in_cb, args.nope_tiles);

            // Phase 3: Tilize ROPE block [8, 64] → 2 tiles
            // Must re-init tilize for different block width (2 tiles vs 8 tiles)
            tilize_uninit(args.receiver_in_cb, args.out_cb);
            tilize_init(args.receiver_in_cb, args.rope_tiles, args.out_cb);

            cb_wait_front(args.receiver_in_cb, args.rope_tiles);
            cb_reserve_back(args.out_cb, args.rope_tiles);
            tilize_block(args.receiver_in_cb, args.rope_tiles, args.out_cb);
            cb_push_back(args.out_cb, args.rope_tiles);
            cb_pop_front(args.receiver_in_cb, args.rope_tiles);
        }
#endif
    };  // class Op

};  // struct CreateQHeads

}  // namespace deepseek_b1_ops
