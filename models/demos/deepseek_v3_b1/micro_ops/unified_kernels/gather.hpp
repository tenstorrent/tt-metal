// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

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

    // Receiver CTArgs: [noc0_num_senders, noc1_num_senders, noc0_receiver_semaphore_id, noc1_receiver_semaphore_id]
    template <
        uint32_t Noc0NumSenders,
        uint32_t Noc1NumSenders,
        uint32_t Noc0ReceiverSemaphoreId,
        uint32_t Noc1ReceiverSemaphoreId>
    struct ReceiverCTArgs {
        static constexpr uint32_t noc0_num_senders = Noc0NumSenders;
        static constexpr uint32_t noc1_num_senders = Noc1NumSenders;
        static constexpr uint32_t noc0_receiver_semaphore_id = Noc0ReceiverSemaphoreId;
        static constexpr uint32_t noc1_receiver_semaphore_id = Noc1ReceiverSemaphoreId;
    };

    // Sender CTArgs: [dest_noc_x, dest_noc_y, data_size_bytes, receiver_semaphore_id]
    template <uint32_t DestNocX, uint32_t DestNocY, uint32_t DataSizeBytes, uint32_t ReceiverSemaphoreId>
    struct SenderCTArgs {
        static constexpr uint32_t dest_noc_x = DestNocX;
        static constexpr uint32_t dest_noc_y = DestNocY;
        static constexpr uint32_t data_size_bytes = DataSizeBytes;
        static constexpr uint32_t receiver_semaphore_id = ReceiverSemaphoreId;
    };

    // Compute CTArgs - not used for gather (dataflow only)
    struct ComputeCTArgs {};

    // ========================================================================
    // Op - the actual operation, templated on CTArgs
    // ========================================================================
    template <typename CTArgs>
    class Op {
    public:
        // ====================================================================
        // Phase-specific RTArgs
        // ====================================================================
        struct ReceiverArgs {};
        struct SenderArgs {
            uint32_t input_data_addr;
            uint32_t receiver_data_addr;
            uint32_t offset;
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
            static_assert(
                CTArgs::data_size_bytes <= NOC_MAX_BURST_SIZE,
                "Data size must be less than or equal to NOC_MAX_BURST_SIZE");
            uint32_t receiver_semaphore_addr = get_semaphore(CTArgs::receiver_semaphore_id);
            const uint64_t dst_noc_coord = get_noc_addr(CTArgs::dest_noc_x, CTArgs::dest_noc_y, 0);
            uint64_t dst_data_noc_addr = dst_noc_coord | (uint64_t)(args.receiver_data_addr + args.offset);
            uint64_t dst_semaphore_noc_addr = dst_noc_coord | (uint64_t)receiver_semaphore_addr;
            noc_async_write_one_packet<true, true>(args.input_data_addr, dst_data_noc_addr, CTArgs::data_size_bytes);
            noc_semaphore_inc<true>(dst_semaphore_noc_addr, 1);
            noc_async_posted_writes_flushed();
#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC (Receiver) - DataMovementProcessor.RISCV_0
            // ================================================================
            uint32_t noc0_receiver_semaphore_addr = get_semaphore(CTArgs::noc0_receiver_semaphore_id);
            uint32_t noc1_receiver_semaphore_addr = get_semaphore(CTArgs::noc1_receiver_semaphore_id);
            volatile tt_l1_ptr uint32_t* noc0_receiver_semaphore_addr_ptr =
                (volatile tt_l1_ptr uint32_t*)noc0_receiver_semaphore_addr;
            volatile tt_l1_ptr uint32_t* noc1_receiver_semaphore_addr_ptr =
                (volatile tt_l1_ptr uint32_t*)noc1_receiver_semaphore_addr;
            noc_semaphore_wait(noc0_receiver_semaphore_addr_ptr, CTArgs::noc0_num_senders);
            noc_semaphore_wait(noc1_receiver_semaphore_addr_ptr, CTArgs::noc1_num_senders);
#endif
        }
    };  // class Op

};  // struct Gather

}  // namespace deepseek_b1_ops
