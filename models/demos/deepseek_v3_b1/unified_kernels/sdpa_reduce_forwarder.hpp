// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// SDPA Reduce-to-All Forwarder Operations
//
// Forwarder core functionality:
// - BRISC (FWD direction): Forwards fabric packets from workers to remote devices
// - NCRISC (BWD direction): Forwards fabric packets from workers to remote devices
// - TRISC: No-op
//
// Design:
// - Single direction-agnostic kernel, same code for BRISC (FWD) and NCRISC (BWD)
// - Direction implicit in fabric connection RT args (src->dst determines direction)
// - BRISC and NCRISC run on same core but access different L1 regions (via buffer_offset)
// - Non-blocking: forwards packets as soon as they arrive (no batching)
//
// TWO-SEMAPHORE DESIGN:
// Each forwarder instance has two semaphores: R1 and R2
// - R1 and R2 use SEPARATE buffer regions to support streaming overlap
// - Interleaved processing: check both semaphores in each poll iteration

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "fabric/fabric_edm_packet_header.hpp"
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include <cstdint>
#endif

namespace deepseek_b1_ops {

// ============================================================================
// SDPA Reduce-to-All Forwarder Operations
// ============================================================================
struct SdpaReduceForwarder {
    // ========================================================================
    // Compile-time args struct
    // ========================================================================
    template <uint32_t slotsPerRound, uint32_t slotSize, uint32_t r2BufferOffset>
    struct CTArgs {
        static constexpr uint32_t slots_per_round = slotsPerRound;
        static constexpr uint32_t slot_size = slotSize;
        static constexpr uint32_t r2_buffer_offset = r2BufferOffset;

        // Avoid undefined behavior for slots_per_round == 32 (shift by width of type).
        static constexpr uint32_t compute_all_sent_mask(uint32_t slots) {
            return (slots == 32) ? 0xFFFF'FFFFu : ((1u << slots) - 1u);
        }
        static constexpr uint32_t all_sent_mask = compute_all_sent_mask(slotsPerRound);
    };

    // ========================================================================
    // Op - unified forwarder operation
    //
    // CT: compile-time args for the forwarder
    // ========================================================================
    template <typename CT>
    class Op {
    public:
        void operator()() {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
            forwarder_impl();
#endif
            // TRISC: no-op
        }

    private:
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
        /**
         * Process ready slots from a semaphore and forward packets.
         * Returns updated sent_mask.
         */
        template <typename FabricConnection>
        FORCE_INLINE uint32_t process_ready_slots(
            volatile tt_l1_ptr uint32_t* sem_ptr,
            uint32_t sent_mask,
            uint32_t buffer_base,
            FabricConnection& fabric_connection) {
            uint32_t sem_value = *sem_ptr;
            uint32_t pending = sem_value & ~sent_mask;

            while (pending != 0) {
                uint32_t slot = __builtin_ctz(pending);
                uint32_t slot_addr = buffer_base + (slot * CT::slot_size);

                // Read actual payload size from packet header
                auto* packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(slot_addr);
                uint32_t actual_packet_size = packet_header->get_payload_size_including_header();

                fabric_connection.wait_for_empty_write_slot();
                fabric_connection.send_payload_flush_non_blocking_from_address(slot_addr, actual_packet_size);

                sent_mask |= (1u << slot);
                pending &= ~(1u << slot);
            }

            return sent_mask;
        }

        void forwarder_impl() {
            static_assert(CT::slots_per_round <= 32, "forwarder supports at most 32 slots per round");

            size_t arg_idx = 0;

            const uint32_t buffer_base = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t buffer_offset = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t r1_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
            const uint32_t r2_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

            const uint32_t my_buffer_base = buffer_base + buffer_offset;

            auto fabric_connection =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
            fabric_connection.open();

            // Interleaved R1/R2 forwarding loop
            volatile tt_l1_ptr uint32_t* r1_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r1_sem_addr);
            volatile tt_l1_ptr uint32_t* r2_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r2_sem_addr);

            const uint32_t r1_buffer_base = my_buffer_base;
            const uint32_t r2_buffer_base = my_buffer_base + CT::r2_buffer_offset;

            uint32_t r1_sent_mask = 0;
            uint32_t r2_sent_mask = 0;

            do {
                invalidate_l1_cache();

                r1_sent_mask = process_ready_slots(r1_sem_ptr, r1_sent_mask, r1_buffer_base, fabric_connection);
                r2_sent_mask = process_ready_slots(r2_sem_ptr, r2_sent_mask, r2_buffer_base, fabric_connection);

            } while (r1_sent_mask != CT::all_sent_mask || r2_sent_mask != CT::all_sent_mask);

            fabric_connection.close();

            noc_async_full_barrier();
        }
#endif  // COMPILE_FOR_NCRISC || COMPILE_FOR_BRISC
    };

};  // struct SdpaReduceForwarder

}  // namespace deepseek_b1_ops
