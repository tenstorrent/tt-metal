// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Forwarder kernel replaces mux with lightweight packet forwarding.
//
// Design:
// - Single direction-agnostic kernel, same code for BRISC (FWD) and NCRISC (BWD)
// - Direction implicit in fabric connection RT args (src→dst determines direction)
// - BRISC and NCRISC run on same core but access different L1 regions (via buffer_offset)
// - Non-blocking: forwards packets as soon as they arrive (no batching)
//
// TWO-SEMAPHORE DESIGN:
// Each forwarder instance has two semaphores: R1 and R2
// - R1 semaphore: tracks R1 packets from half the workers (Type A for FWD, Type B for BWD)
// - R2 semaphore: tracks R2 packets from other half (Type B for FWD, Type A for BWD)
// - R1 and R2 use SEPARATE buffer regions to support streaming overlap
// - Interleaved processing: check both semaphores in each poll iteration
//
// Memory layout per forwarder kernel instance:
// - R1 buffer region: [slot0][slot1]...[slotN-1] at buffer_base
// - R2 buffer region: [slot0][slot1]...[slotN-1] at buffer_base + r2_buffer_offset
// - Separate regions allow R2 to start while R1 forwarding is still in progress
//
// Semaphore protocol (bit-packed, 32-bit each):
// - Each semaphore tracks slots_per_round slots
// - Workers signal with (1 << slot_idx)
// - Forwarder polls both semaphores and forwards any ready slots
// - Maximum 32 slots per semaphore (limited by 32-bit width)
//
// Workflow:
// 1. Poll both r1_sem and r2_sem
// 2. For each ready bit not yet sent, forward the packet
// 3. Repeat until both semaphores fully processed

#include "fabric/fabric_edm_packet_header.hpp"
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include <cstdint>

// =============================================================================
// Compile-time arguments
// =============================================================================
static constexpr uint32_t slots_per_round = get_compile_time_arg_val(0);   // Slots per semaphore (R1 or R2)
static constexpr uint32_t slot_size = get_compile_time_arg_val(1);         // Max bytes per slot
static constexpr uint32_t r2_buffer_offset = get_compile_time_arg_val(2);  // Offset for R2 buffer region

// Avoid undefined behavior for slots_per_round == 32 (shift by width of type).
static constexpr uint32_t compute_all_sent_mask(uint32_t slots) {
    return (slots == 32) ? 0xFFFF'FFFFu : ((1u << slots) - 1u);
}
static constexpr uint32_t all_sent_mask = compute_all_sent_mask(slots_per_round);

// Maximum slots is 32 (one bit per slot in 32-bit semaphore)
static_assert(slots_per_round <= 32, "forwarder supports at most 32 slots per round (limited by 32-bit semaphore)");

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
        uint32_t slot_addr = buffer_base + (slot * slot_size);

        // Read actual payload size from packet header (supports variable chunk sizes)
        auto* packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(slot_addr);
        uint32_t actual_packet_size = packet_header->get_payload_size_including_header();

        fabric_connection.wait_for_empty_write_slot();
        fabric_connection.send_payload_flush_non_blocking_from_address(slot_addr, actual_packet_size);

        sent_mask |= (1u << slot);
        pending &= ~(1u << slot);
    }

    return sent_mask;
}

void kernel_main() {
    size_t arg_idx = 0;

    const uint32_t buffer_base = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t buffer_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r1_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t r2_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    const uint32_t my_buffer_base = buffer_base + buffer_offset;

    auto fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
    fabric_connection.open();

    // =========================================================================
    // Interleaved R1/R2 forwarding loop
    // =========================================================================
    // Poll both semaphores and forward packets as they're ready.
    // R1 and R2 use separate buffer regions to support streaming overlap.

    volatile tt_l1_ptr uint32_t* r1_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r1_sem_addr);
    volatile tt_l1_ptr uint32_t* r2_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r2_sem_addr);

    // R1 and R2 buffer bases are offset to prevent overlap during streaming
    const uint32_t r1_buffer_base = my_buffer_base;
    const uint32_t r2_buffer_base = my_buffer_base + r2_buffer_offset;

    uint32_t r1_sent_mask = 0;
    uint32_t r2_sent_mask = 0;

    do {
        invalidate_l1_cache();

        // Process R1 slots (at r1_buffer_base)
        r1_sent_mask = process_ready_slots(r1_sem_ptr, r1_sent_mask, r1_buffer_base, fabric_connection);

        // Process R2 slots (at r2_buffer_base, separate region)
        r2_sent_mask = process_ready_slots(r2_sem_ptr, r2_sent_mask, r2_buffer_base, fabric_connection);

    } while (r1_sent_mask != all_sent_mask || r2_sent_mask != all_sent_mask);

    fabric_connection.close();

    noc_async_full_barrier();
}
