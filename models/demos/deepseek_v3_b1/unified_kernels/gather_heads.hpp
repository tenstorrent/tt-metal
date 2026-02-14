// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// Gather Heads micro-op
//
// Gathers data from 12x8 sender cores to 4x2 receiver cores.
// Each sender row maps to a different receiver core:
//   Row 0 → (0,1), Row 1 → (1,1), Row 2 → (2,1), Row 3 → (3,1)
//   Row 4 → (0,2), Row 5 → (1,2), Row 6 → (2,2), Row 7 → (3,2)
//
// Receiver layout (8 heads per core, each head = 576 elements):
//   Head 0: qnope[col=0] (512) + qrope[col=8, head0] (64)
//   Head 1: qnope[col=1] (512) + qrope[col=8, head1] (64)
//   ...
//   Head 7: qnope[col=7] (512) + qrope[col=11, head1] (64)
//
// Sender offsets (in elements):
//   - Qnope col X (0-7): offset = X * 576 (sends 512 elements)
//   - Qrope col X (8-11): sends 2 chunks of 64 elements each:
//     - Head 0: offset = 512 + 2*(X-8)*576
//     - Head 1: offset = 512 + (2*(X-8)+1)*576
// ============================================================================
struct GatherHeads {
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
        // Semaphore
        uint32_t receiver_semaphore_id;
        // Target NOC coordinates (8 rows, packed: lower 16 bits = x, upper 16 bits = y)
        uint32_t target_noc_coords[8];
        // Runtime arg - destination address
        uint32_t receiver_data_addr;
    };

    // Receiver args
    struct ReceiverArgs {
        uint32_t receiver_semaphore_id;
        uint32_t num_senders;  // Total expected semaphore count
        uint32_t out_cb;
        uint32_t dst_num_pages;
    };

    // Compute args (TRISC) - not used for gather (dataflow only)
    struct ComputeArgs {};

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

        // Overload for compute args (no-op)
        void operator()([[maybe_unused]] const ComputeArgs& args) {
            // TRISC: No-op (gather is dataflow only)
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

            // Get target NOC coordinates based on row (unpacked from uint32)
            uint32_t packed_coords = args.target_noc_coords[my_row];
            uint32_t target_noc_x = packed_coords & 0xFFFF;          // Lower 16 bits
            uint32_t target_noc_y = (packed_coords >> 16) & 0xFFFF;  // Upper 16 bits

            // Get receiver semaphore address
            uint32_t receiver_semaphore_addr = get_semaphore(args.receiver_semaphore_id);
            const uint64_t dst_noc_coord = get_noc_addr(target_noc_x, target_noc_y, 0);
            uint64_t dst_semaphore_noc_addr = dst_noc_coord | (uint64_t)receiver_semaphore_addr;

            // Wait for source CB data to be ready
            cb_wait_front(src_cb, args.src_num_pages);

            // Get source address from CB
            uint32_t src_addr = get_read_ptr(src_cb);

            if (is_qnope_core) {
                // Qnope core: sends 512 elements to offset = col * head_stride
                uint32_t dst_offset = my_col * args.head_stride_bytes;
                uint64_t dst_data_noc_addr = dst_noc_coord | (uint64_t)(args.receiver_data_addr + dst_offset);
                noc_async_write(src_addr, dst_data_noc_addr, args.qnope_data_size_bytes);
            } else {
                // Qrope core: sends 2 chunks of 64 elements each
                uint32_t qrope_col = my_col - args.qnope_cols;
                uint32_t dst_offset0 = args.qnope_data_size_bytes + 2 * qrope_col * args.head_stride_bytes;
                uint64_t dst_data_noc_addr0 = dst_noc_coord | (uint64_t)(args.receiver_data_addr + dst_offset0);
                uint32_t dst_offset1 = args.qnope_data_size_bytes + (2 * qrope_col + 1) * args.head_stride_bytes;
                uint64_t dst_data_noc_addr1 = dst_noc_coord | (uint64_t)(args.receiver_data_addr + dst_offset1);
                // We should use one packet APIs here if we assert/know ahead of time the txn size
                // Or if txn size is a compile time arg to pass it here to automatically select to use one packet
                noc_async_write<NOC_MAX_BURST_SIZE + 1, true, /*posted=*/true>(
                    src_addr, dst_data_noc_addr0, args.qrope_head_size_bytes);
                noc_async_write<NOC_MAX_BURST_SIZE + 1, true, /*posted=*/true>(
                    src_addr + args.qrope_head_size_bytes, dst_data_noc_addr1, args.qrope_head_size_bytes);
            }

            noc_semaphore_inc(dst_semaphore_noc_addr, 1);

            noc_async_posted_writes_flushed();

            // Pop source CB after sending
            if constexpr (pop_src) {
                cb_pop_front(src_cb, args.src_num_pages);
            }
            // This also guarantees the previous posted writes have landed since we're on the same VC
            noc_async_atomic_barrier();
        }

        FORCE_INLINE void receiver_impl(const ReceiverArgs& args) {
            // Wait for all senders
            uint32_t semaphore_addr = get_semaphore(args.receiver_semaphore_id);
            volatile tt_l1_ptr uint32_t* semaphore_ptr = (volatile tt_l1_ptr uint32_t*)semaphore_addr;

            // Reserve space in output CB if using CB output
            if constexpr (use_cb_output) {
                cb_reserve_back(args.out_cb, args.dst_num_pages);
            }

            if (args.num_senders > 0) {
                noc_semaphore_wait(semaphore_ptr, args.num_senders);
            }
            noc_semaphore_set(semaphore_ptr, 0);

            // Push to output CB if using CB output
            if constexpr (use_cb_output) {
                cb_push_back(args.out_cb, args.dst_num_pages);
            }
        }

#endif
    };  // class Op

};  // struct GatherHeads

}  // namespace deepseek_b1_ops
