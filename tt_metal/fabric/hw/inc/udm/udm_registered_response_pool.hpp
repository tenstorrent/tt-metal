// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "debug/assert.h"
#include "dataflow_api.h"
#include "noc_parameters.h"
#include "fabric/fabric_edm_packet_header.hpp"

namespace tt::tt_fabric::udm {

/**
 * @brief Registered response entry for tracking pending write/read responses
 *
 * This struct contains all information needed to send a response back to the requestor
 * after the NOC operation completes. For writes, this includes acknowledgment info.
 * For reads, this includes source location and memory pool slot tracking.
 */
struct RegisteredResponse {
    uint8_t noc_send_type;
    uint8_t risc_id;
    uint8_t mux_index;

    // Common fields from UDMControlFields (excluding initial_direction)
    uint8_t src_chip_id;
    uint16_t src_mesh_id;
    uint8_t src_noc_x;
    uint8_t src_noc_y;
    uint8_t transaction_id;

    // Memory pool tracking for reads (bytes-based, no slot counting)
    uint32_t bytes_remaining;  // Bytes in memory pool waiting to be sent

    // Padding to reach 32 bytes
    uint8_t padding[3];

    // Read-specific fields
    uint32_t src_l1_address;
    uint32_t bytes_to_allocate;   // Bytes remaining to allocate from source
    uint64_t read_noc_address;    // Where to read from next (updated after allocation)

    // Helper to populate fields from header
    // For reads, also pass noc_addr; for writes, it's ignored
    template <tt::tt_fabric::NocSendType noc_send_type, typename HeaderType>
    FORCE_INLINE void populate_from_header(
        const volatile HeaderType& header, uint8_t mux_index, uint64_t noc_addr = 0) volatile {
        this->noc_send_type = noc_send_type;
        this->risc_id = header.risc_id;
        this->mux_index = mux_index;

        src_chip_id = header.src_chip_id;
        src_mesh_id = header.src_mesh_id;
        src_noc_x = header.src_noc_x;
        src_noc_y = header.src_noc_y;
        transaction_id = header.transaction_id;

        // For reads, populate read-specific fields (compile-time check)
        if constexpr (noc_send_type == tt::tt_fabric::NocSendType::NOC_UNICAST_READ) {
            src_l1_address = header.src_l1_address;
            bytes_to_allocate = header.size_bytes;
            read_noc_address = noc_addr;
            bytes_remaining = 0;
        }
    }

    // Check if there are bytes in memory pool to send
    FORCE_INLINE bool has_data_to_send() const volatile { return bytes_remaining > 0; }

    // Check if this is the last slot to send (for fused atomic on final packet)
    // True when: all bytes allocated AND only one slot's worth (or less) remaining
    FORCE_INLINE bool is_last_send(uint32_t slot_size) const volatile {
        return bytes_to_allocate == 0 && bytes_remaining <= slot_size;
    }

    // Called after allocating: advance read address, update byte counters
    FORCE_INLINE void complete_allocation(uint32_t bytes_allocated) volatile {
        read_noc_address += bytes_allocated;
        bytes_to_allocate -= bytes_allocated;
        bytes_remaining += bytes_allocated;
    }

    // Called after sending ONE slot: advance dest address, decrement bytes
    // Returns true if all data has been sent
    FORCE_INLINE bool complete_send(uint32_t slot_size) volatile {
        uint32_t bytes_sent = std::min((uint32_t)bytes_remaining, slot_size);

        src_l1_address += bytes_sent;
        bytes_remaining -= bytes_sent;

        return bytes_remaining == 0 && bytes_to_allocate == 0;
    }
} __attribute__((packed));

static_assert(sizeof(RegisteredResponse) == 32, "RegisteredResponse must be exactly 32 bytes");

/**
 * @brief Pool for managing registered responses that are pending completion
 *
 * This pool manages a circular buffer of RegisteredResponse entries in L1 memory.
 * Unlike UDMMemoryPool which stores actual data in an array, this pool only maintains
 * read/write indices and accesses the L1 memory directly via pointers, as the number
 * of slots can be very large (thousands).
 */
template <uint32_t BaseAddress, uint32_t NumSlots>
class RegisteredResponsePool {
private:
    uint32_t wr_slot_idx_ = 0;
    uint32_t rd_slot_idx_ = 0;
    uint32_t num_used_slots_ = 0;

    static constexpr uint32_t SLOT_SIZE = sizeof(RegisteredResponse);

    // Get pointer to response slot at index
    FORCE_INLINE volatile RegisteredResponse* get_slot_ptr(uint32_t idx) const {
        uint32_t addr = BaseAddress + idx * SLOT_SIZE;
        return reinterpret_cast<volatile RegisteredResponse*>(addr);
    }

    // Advance slot index by 1 with wrap
    FORCE_INLINE uint32_t advance_slot_idx(uint32_t idx) const {
        idx++;
        if (idx >= NumSlots) {
            idx = 0;
        }
        return idx;
    }

public:
    // Constructor
    RegisteredResponsePool() = default;

    // Check if pool has space for a new entry
    FORCE_INLINE bool has_space() const { return num_used_slots_ < NumSlots; }

    // Check if pool is empty
    FORCE_INLINE bool is_empty() const { return num_used_slots_ == 0; }

    // Get pointer to current write slot (for registering new response)
    FORCE_INLINE volatile RegisteredResponse* get_unregistered_slot() {
        ASSERT(has_space());
        return get_slot_ptr(wr_slot_idx_);
    }

    // Get pointer to current read slot (for processing next response)
    FORCE_INLINE volatile RegisteredResponse* get_registered_slot() const {
        ASSERT(!is_empty());
        return get_slot_ptr(rd_slot_idx_);
    }

    // Register a new response (allocate write slot)
    FORCE_INLINE void register_response() {
        ASSERT(has_space());
        wr_slot_idx_ = advance_slot_idx(wr_slot_idx_);
        num_used_slots_++;
    }

    // Unregister a response (deallocate read slot)
    FORCE_INLINE void unregister_response() {
        ASSERT(!is_empty());
        rd_slot_idx_ = advance_slot_idx(rd_slot_idx_);
        num_used_slots_--;
    }
};

}  // namespace tt::tt_fabric::udm
