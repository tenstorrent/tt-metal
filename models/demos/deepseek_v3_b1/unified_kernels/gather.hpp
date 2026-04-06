// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// Gather micro-op
//
// Gathers data from multiple sender cores to a single receiver core.
// Default: Sender on NCRISC, Receiver on BRISC.
// ReceiverOnNCRISC mode: Both sender and receiver on NCRISC, BRISC is no-op.
//   Use when sender and receiver share a core (e.g. dkv_gather where
//   kv_rmsnorm_core is also a knope sender core).
//
// CB States:
//   Sender (NCRISC):
//     - Waits: src_cb (src_num_pages)
//     - Pops: src_cb (src_num_pages) if pop_src=true
//   Receiver (BRISC, or NCRISC when ReceiverOnNCRISC=true):
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
    // Runtime args structs
    // ========================================================================

    struct ReceiverArgs {
        uint32_t noc0_num_senders;
        uint32_t noc1_num_senders;
        uint32_t noc0_receiver_semaphore_addr;
        uint32_t noc1_receiver_semaphore_addr;
        uint32_t dst_cb;
        uint32_t dst_num_pages;
    };

    struct SenderArgs {
        uint32_t dest_noc_x;
        uint32_t dest_noc_y;
        uint32_t data_size_bytes;
        uint32_t receiver_semaphore_addr;
        uint32_t src_cb;
        uint32_t src_num_pages;
        uint32_t sender_grid_start_x;
        uint32_t sender_grid_start_y;
        uint32_t sender_grid_end_x;
        uint32_t sender_grid_end_y;
        uint32_t row_major;
        uint32_t receiver_data_addr;
        uint32_t sender_idx;  // Per-core sender index (only used if UsePerCoreSenderIdx=true)
    };

    // Unified dataflow args: both BRISC and NCRISC get the full set so the
    // sender/receiver impl can be placed on either RISC without restructuring.
    struct DMArgs {
        SenderArgs sender;
        ReceiverArgs receiver;
    };

    // Compute args (TRISC) - not used for gather (dataflow only)
    struct ComputeArgs {};

    using RTArgs = unified_kernels::SelectByRISCV<DMArgs, DMArgs, ComputeArgs>;

    // ========================================================================
    // Op - the actual operation
    //
    // IsSenderCore: compile-time flag to distinguish sender vs receiver cores
    // IsReceiverCore: compile-time flag for receiver cores
    // pop_src: whether to pop the source CB after sending
    // UsePerCoreSenderIdx: compile-time flag for scattered vs grid-based indexing
    // ReceiverOnNCRISC: when true, receiver logic runs on NCRISC instead of BRISC.
    //   Use for gathers where sender and receiver share a core (e.g. dkv_gather).
    //   NCRISC does send-then-receive; BRISC is no-op.
    // ========================================================================
    template <
        bool IsSenderCore,
        bool IsReceiverCore,
        bool pop_src,
        bool UsePerCoreSenderIdx = false,
        bool ReceiverOnNCRISC = false>
    class Op {
    public:
        void operator()(const RTArgs& args) { impl(args); }

    private:
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
        static void sender_impl(const SenderArgs& s) {
            ASSERT(s.data_size_bytes > 0);
            uint32_t core_index;
            if constexpr (UsePerCoreSenderIdx) {
                core_index = s.sender_idx;
            } else {
                core_index = unified_kernels::linear_id_in_grid<true>(
                    s.sender_grid_start_x, s.sender_grid_start_y, s.sender_grid_end_x, s.sender_grid_end_y);
            }
            uint32_t offset = core_index * s.data_size_bytes;

            const uint64_t dst_noc_coord = get_noc_addr(s.dest_noc_x, s.dest_noc_y, 0);
            uint64_t dst_data_noc_addr = dst_noc_coord | (uint64_t)(s.receiver_data_addr + offset);
            uint64_t dst_semaphore_noc_addr = dst_noc_coord | (uint64_t)s.receiver_semaphore_addr;

            cb_wait_front(s.src_cb, s.src_num_pages);

            uint32_t input_data_addr = get_read_ptr(s.src_cb);
            noc_async_write_one_packet<true, true>(input_data_addr, dst_data_noc_addr, s.data_size_bytes);
            // BH does not support posted atomics due to a bug
            noc_semaphore_inc(dst_semaphore_noc_addr, 1);

            if constexpr (pop_src) {
                noc_async_posted_writes_flushed();
                cb_pop_front(s.src_cb, s.src_num_pages);
            }
            noc_async_atomic_barrier();
        }

        static void receiver_impl(const ReceiverArgs& r) {
            ASSERT(r.noc0_num_senders > 0 || r.noc1_num_senders > 0);
            volatile tt_l1_ptr uint32_t* noc0_receiver_semaphore_addr_ptr =
                (volatile tt_l1_ptr uint32_t*)r.noc0_receiver_semaphore_addr;

            cb_reserve_back(r.dst_cb, r.dst_num_pages);
            noc_semaphore_wait(noc0_receiver_semaphore_addr_ptr, r.noc0_num_senders);
            noc_semaphore_set(noc0_receiver_semaphore_addr_ptr, 0);

            if (r.noc1_num_senders > 0) {
                volatile tt_l1_ptr uint32_t* noc1_receiver_semaphore_addr_ptr =
                    (volatile tt_l1_ptr uint32_t*)r.noc1_receiver_semaphore_addr;
                noc_semaphore_wait(noc1_receiver_semaphore_addr_ptr, r.noc1_num_senders);
                noc_semaphore_set(noc1_receiver_semaphore_addr_ptr, 0);
            }

            cb_push_back(r.dst_cb, r.dst_num_pages);
        }
#endif

        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC: Sender always, Receiver when ReceiverOnNCRISC=true
            // ================================================================
            if constexpr (IsSenderCore) {
                sender_impl(args.sender);
            }
            if constexpr (ReceiverOnNCRISC && IsReceiverCore) {
                receiver_impl(args.receiver);
            }
#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC: Receiver when ReceiverOnNCRISC=false, no-op otherwise
            // ================================================================
            if constexpr (!ReceiverOnNCRISC && IsReceiverCore) {
                receiver_impl(args.receiver);
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
