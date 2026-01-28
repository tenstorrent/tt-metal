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
// Gather micro-op
//
// Gathers data from multiple sender cores to a single receiver core.
// Receiver runs on BRISC, Sender runs on NCRISC.
//
// CB States:
//   NCRISC (Sender):
//     - Waits: src_cb (src_num_pages)
//     - Pops: src_cb (src_num_pages) if pop_src=true
//   BRISC (Receiver):
//     - Reserves: dst_cb (dst_num_pages)
//     - Pushes: dst_cb (dst_num_pages)
//   TRISC: No-op
//
// Semaphore States (separate semaphore per NOC):
//   Sender: Assumes both noc0/noc1 semaphores start at 0, increments one after sending
//           (which NOC depends on sender's position in grid)
//   Receiver: Waits for noc0_semaphore == noc0_num_senders and noc1_semaphore == noc1_num_senders,
//             then resets both to 0
//
// Note: Sender assumes that receiver's dst_cb is ready to receive at the beginning of NCRISC execution.
// ============================================================================
struct Gather {
    // ========================================================================
    // Runtime args structs - different layout per RISC
    // ========================================================================

    // Receiver args (BRISC): [noc0_num_senders, noc1_num_senders, noc0_receiver_semaphore_id,
    //                         noc1_receiver_semaphore_id, dst_cb, dst_num_pages]
    struct ReceiverArgs {
        uint32_t noc0_num_senders;
        uint32_t noc1_num_senders;
        uint32_t noc0_receiver_semaphore_id;
        uint32_t noc1_receiver_semaphore_id;
        uint32_t dst_cb;
        uint32_t dst_num_pages;
    };

    // Sender args (NCRISC): [dest_noc_x, dest_noc_y, data_size_bytes, receiver_semaphore_id,
    //                        src_cb, src_num_pages, sender_grid_start_x, sender_grid_start_y,
    //                        sender_grid_end_x, sender_grid_end_y, row_major, receiver_data_addr]
    struct SenderArgs {
        uint32_t dest_noc_x;
        uint32_t dest_noc_y;
        uint32_t data_size_bytes;
        uint32_t receiver_semaphore_id;
        uint32_t src_cb;
        uint32_t src_num_pages;
        uint32_t sender_grid_start_x;
        uint32_t sender_grid_start_y;
        uint32_t sender_grid_end_x;
        uint32_t sender_grid_end_y;
        uint32_t row_major;
        uint32_t receiver_data_addr;
    };

    // Compute args (TRISC) - not used for gather (dataflow only)
    struct ComputeArgs {};

    // Note: For gather, NCRISC=Sender, BRISC=Receiver
    using RTArgs = unified_kernels::SelectByRISCV<SenderArgs, ReceiverArgs, ComputeArgs>;

    // ========================================================================
    // Op - the actual operation
    //
    // IsSenderCore: compile-time flag to distinguish sender vs receiver cores
    // IsReceiverCore: compile-time flag for receiver cores
    // pop_src: whether to pop the source CB after sending
    // ========================================================================
    template <bool IsSenderCore, bool IsReceiverCore, bool pop_src>
    class Op {
    public:
        void operator()(const RTArgs& args) { impl(args); }

    private:
        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC (Sender) - DataMovementProcessor.RISCV_1
            // ================================================================
            if constexpr (IsSenderCore) {
                // Wait for source CB data to be ready
                cb_wait_front(args.src_cb, args.src_num_pages);

                // Get source address from CB
                uint32_t input_data_addr = get_read_ptr(args.src_cb);

                // Compute per-core offset based on logical core coordinates
                // Note: my_logical_x_/y_ are global variables set by firmware
                uint32_t core_index = unified_kernels::linear_id_in_grid<true>(
                    args.sender_grid_start_x, args.sender_grid_start_y, args.sender_grid_end_x, args.sender_grid_end_y);
                uint32_t offset = core_index * args.data_size_bytes;

                uint32_t receiver_semaphore_addr = get_semaphore(args.receiver_semaphore_id);
                const uint64_t dst_noc_coord = get_noc_addr(args.dest_noc_x, args.dest_noc_y, 0);
                uint64_t dst_data_noc_addr = dst_noc_coord | (uint64_t)(args.receiver_data_addr + offset);
                uint64_t dst_semaphore_noc_addr = dst_noc_coord | (uint64_t)receiver_semaphore_addr;
                noc_async_write_one_packet<true, true>(input_data_addr, dst_data_noc_addr, args.data_size_bytes);
                noc_semaphore_inc<true>(dst_semaphore_noc_addr, 1);
                noc_async_posted_writes_flushed();

                // Pop the source CB after sending
                if constexpr (pop_src) {
                    cb_pop_front(args.src_cb, args.src_num_pages);
                }
            }
#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC (Receiver) - DataMovementProcessor.RISCV_0
            // ================================================================
            if constexpr (IsReceiverCore) {
                // Reserve space in destination CB
                cb_reserve_back(args.dst_cb, args.dst_num_pages);

                uint32_t noc0_receiver_semaphore_addr = get_semaphore(args.noc0_receiver_semaphore_id);
                uint32_t noc1_receiver_semaphore_addr = get_semaphore(args.noc1_receiver_semaphore_id);
                volatile tt_l1_ptr uint32_t* noc0_receiver_semaphore_addr_ptr =
                    (volatile tt_l1_ptr uint32_t*)noc0_receiver_semaphore_addr;
                volatile tt_l1_ptr uint32_t* noc1_receiver_semaphore_addr_ptr =
                    (volatile tt_l1_ptr uint32_t*)noc1_receiver_semaphore_addr;
                noc_semaphore_wait(noc0_receiver_semaphore_addr_ptr, args.noc0_num_senders);
                noc_semaphore_wait(noc1_receiver_semaphore_addr_ptr, args.noc1_num_senders);
                noc_semaphore_set(noc0_receiver_semaphore_addr_ptr, 0);
                noc_semaphore_set(noc1_receiver_semaphore_addr_ptr, 0);

                // Push to destination CB after data arrived
                cb_push_back(args.dst_cb, args.dst_num_pages);
            }
#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC - No-op (gather is dataflow only)
            // ================================================================
#endif
        }
    };  // class Op

};  // struct Gather

}  // namespace deepseek_b1_ops
