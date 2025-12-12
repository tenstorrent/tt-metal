// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#include "models/demos/deepseek_v3_b1/fused_ops/pre_sdpa/kernels/unified_core_descriptor.hpp"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "dataflow_api.h"
#endif

#if defined(COMPILE_FOR_NCRISC)
#include "hostdevcommon/common_values.hpp"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// Gather micro-op
//
// Gathers data from multiple sender cores to a single receiver core.
// Receiver runs on BRISC, Sender runs on NCRISC.
// ============================================================================
struct Gather {
    // ========================================================================
    // Compile-time args structs - different layout per RISC
    // ========================================================================

    // Receiver CTArgs: [noc0_num_senders, noc1_num_senders, noc0_receiver_semaphore_id, noc1_receiver_semaphore_id,
    //                   dst_cb, dst_num_pages]
    template <
        uint32_t Noc0NumSenders,
        uint32_t Noc1NumSenders,
        uint32_t Noc0ReceiverSemaphoreId,
        uint32_t Noc1ReceiverSemaphoreId,
        uint32_t DstCb,
        uint32_t DstNumPages>
    struct ReceiverCTArgs {
        static constexpr uint32_t noc0_num_senders = Noc0NumSenders;
        static constexpr uint32_t noc1_num_senders = Noc1NumSenders;
        static constexpr uint32_t noc0_receiver_semaphore_id = Noc0ReceiverSemaphoreId;
        static constexpr uint32_t noc1_receiver_semaphore_id = Noc1ReceiverSemaphoreId;
        static constexpr uint32_t dst_cb = DstCb;
        static constexpr uint32_t dst_num_pages = DstNumPages;
    };

    // Sender CTArgs: [dest_noc_x, dest_noc_y, data_size_bytes, receiver_semaphore_id,
    //                 src_cb, src_num_pages, sender_grid_start_x, sender_grid_start_y, sender_grid_end_x,
    //                 sender_grid_end_y]
    // Note: grid coordinates are in LOGICAL space (not NOC)
    template <
        uint32_t DestNocX,
        uint32_t DestNocY,
        uint32_t DataSizeBytes,
        uint32_t ReceiverSemaphoreId,
        uint32_t SrcCb,
        uint32_t SrcNumPages,
        uint32_t SenderGridStartX,
        uint32_t SenderGridStartY,
        uint32_t SenderGridEndX,
        uint32_t SenderGridEndY>
    struct SenderCTArgs {
        static constexpr uint32_t dest_noc_x = DestNocX;
        static constexpr uint32_t dest_noc_y = DestNocY;
        static constexpr uint32_t data_size_bytes = DataSizeBytes;
        static constexpr uint32_t receiver_semaphore_id = ReceiverSemaphoreId;
        static constexpr uint32_t src_cb = SrcCb;
        static constexpr uint32_t src_num_pages = SrcNumPages;
        static constexpr uint32_t sender_grid_start_x = SenderGridStartX;
        static constexpr uint32_t sender_grid_start_y = SenderGridStartY;
        static constexpr uint32_t sender_grid_end_x = SenderGridEndX;
        static constexpr uint32_t sender_grid_end_y = SenderGridEndY;
    };

    // Compute CTArgs - not used for gather (dataflow only)
    struct ComputeCTArgs {};

    // ========================================================================
    // Op - the actual operation, templated on CTArgs and role
    //
    // IsSenderCore: compile-time flag to distinguish sender vs receiver cores
    // IsReceiverCore: compile-time flag for receiver cores
    // ========================================================================
    template <typename CTArgs, bool IsSenderCore = false, bool IsReceiverCore = false>
    class Op {
    public:
        // ====================================================================
        // Phase-specific RTArgs
        // ====================================================================
        struct ReceiverArgs {};
        struct SenderArgs {
            uint32_t receiver_data_addr;
        };
        struct ComputeArgs {};

        // Note: For gather, NCRISC=Sender, BRISC=Receiver
        using RTArgs = SelectByRISCV<SenderArgs, ReceiverArgs, ComputeArgs>;

        void operator()(const RTArgs& args = {}) { impl(args); }

    private:
        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC (Sender) - DataMovementProcessor.RISCV_1
            // ================================================================
            if constexpr (IsSenderCore) {
                // Wait for source CB data to be ready
                cb_wait_front(CTArgs::src_cb, CTArgs::src_num_pages);

                // Get source address from CB
                uint32_t input_data_addr = get_read_ptr(CTArgs::src_cb);

                // Compute per-core offset based on logical core coordinates
                pre_sdpa::UnifiedCoreDescriptor core;
                uint32_t core_index = core.linear_id_in_grid(
                    CTArgs::sender_grid_start_x,
                    CTArgs::sender_grid_start_y,
                    CTArgs::sender_grid_end_x,
                    CTArgs::sender_grid_end_y);
                uint32_t offset = core_index * CTArgs::data_size_bytes;

                static_assert(
                    CTArgs::data_size_bytes <= NOC_MAX_BURST_SIZE,
                    "Data size must be less than or equal to NOC_MAX_BURST_SIZE");
                uint32_t receiver_semaphore_addr = get_semaphore(CTArgs::receiver_semaphore_id);
                const uint64_t dst_noc_coord = get_noc_addr(CTArgs::dest_noc_x, CTArgs::dest_noc_y, 0);
                uint64_t dst_data_noc_addr = dst_noc_coord | (uint64_t)(args.receiver_data_addr + offset);
                uint64_t dst_semaphore_noc_addr = dst_noc_coord | (uint64_t)receiver_semaphore_addr;
                noc_async_write_one_packet<true, true>(input_data_addr, dst_data_noc_addr, CTArgs::data_size_bytes);
                noc_semaphore_inc<true>(dst_semaphore_noc_addr, 1);
                noc_async_posted_writes_flushed();

                // Pop the source CB after sending
                cb_pop_front(CTArgs::src_cb, CTArgs::src_num_pages);
            }
#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC (Receiver) - DataMovementProcessor.RISCV_0
            // ================================================================
            if constexpr (IsReceiverCore) {
                // Reserve space in destination CB
                cb_reserve_back(CTArgs::dst_cb, CTArgs::dst_num_pages);

                uint32_t noc0_receiver_semaphore_addr = get_semaphore(CTArgs::noc0_receiver_semaphore_id);
                uint32_t noc1_receiver_semaphore_addr = get_semaphore(CTArgs::noc1_receiver_semaphore_id);
                volatile tt_l1_ptr uint32_t* noc0_receiver_semaphore_addr_ptr =
                    (volatile tt_l1_ptr uint32_t*)noc0_receiver_semaphore_addr;
                volatile tt_l1_ptr uint32_t* noc1_receiver_semaphore_addr_ptr =
                    (volatile tt_l1_ptr uint32_t*)noc1_receiver_semaphore_addr;
                noc_semaphore_wait(noc0_receiver_semaphore_addr_ptr, CTArgs::noc0_num_senders);
                noc_semaphore_wait(noc1_receiver_semaphore_addr_ptr, CTArgs::noc1_num_senders);

                // Push to destination CB after data arrived
                cb_push_back(CTArgs::dst_cb, CTArgs::dst_num_pages);
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
