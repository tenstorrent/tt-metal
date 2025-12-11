// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "debug/assert.h"
#include "dataflow_api.h"
#include "noc_parameters.h"
#include "udm_registered_response_pool.hpp"

namespace tt::tt_fabric::udm {

// Slot-based circular buffer memory pool for UDM read responses
// All parameters are compile-time template arguments
template <uint32_t BaseAddress, uint32_t SlotSize, uint32_t NumSlots>
class UDMMemoryPool {
    static_assert(SlotSize % DRAM_ALIGNMENT == 0, "SlotSize must be DRAM aligned");

private:
    uint32_t slot_addrs_[NumSlots];

    uint32_t wr_slot_idx_ = 0;
    uint32_t rd_slot_idx_ = 0;
    uint32_t num_used_slots_ = 0;

    // Advance slot index by 1 with wrap
    FORCE_INLINE uint32_t advance_slot_idx(uint32_t idx) const {
        idx++;
        if (idx >= NumSlots) {
            idx = 0;
        }
        return idx;
    }

    // Advance slot index by n with wrap
    FORCE_INLINE uint32_t advance_slot_idx_n(uint32_t idx, uint32_t n) const { return (idx + n) % NumSlots; }

public:
    // Constructor - pre-caches all slot addresses
    UDMMemoryPool() {
        for (uint32_t i = 0; i < NumSlots; i++) {
            slot_addrs_[i] = BaseAddress + i * SlotSize;
        }
    }

    // Get number of available slots
    FORCE_INLINE uint32_t get_num_available_slots() const { return NumSlots - num_used_slots_; }

    // Get slot size
    FORCE_INLINE constexpr uint32_t get_slot_size() const { return SlotSize; }

    // Get slot address by index
    FORCE_INLINE uint32_t get_slot_addr(uint32_t idx) const { return slot_addrs_[idx]; }

    // Get current read slot index
    FORCE_INLINE uint32_t get_rd_slot_idx() const { return rd_slot_idx_; }

    // Get next slot index (with wrap)
    FORCE_INLINE uint32_t get_next_slot_idx(uint32_t idx) const { return advance_slot_idx(idx); }

    // Allocate slots for a read response - bytes based
    // Performs partial allocation: allocates min(bytes_needed, available_space).
    // Reads from response->read_noc_address (auto-advanced), updates bytes_remaining and bytes_to_allocate.
    // Callers should check response->bytes_to_allocate to determine if more allocation is needed.
    FORCE_INLINE void cb_allocate_and_fill_slots(volatile RegisteredResponse* response) {
        uint32_t bytes_to_allocate = response->bytes_to_allocate;
        uint32_t available_slots = get_num_available_slots();
        bool can_allocate = (bytes_to_allocate != 0) && (available_slots != 0);
        if (can_allocate) {
            // Partial allocation: allocate as much as available space allows
            uint32_t bytes_to_read = std::min(bytes_to_allocate, available_slots * SlotSize);
            uint32_t bytes_allocated = bytes_to_read;

            // Read from current address (already points to next unread location)
            uint64_t src_addr = response->read_noc_address;

            // Read full slots
            while (bytes_to_read > SlotSize) {
                noc_async_read(src_addr, slot_addrs_[wr_slot_idx_], SlotSize);
                src_addr += SlotSize;
                bytes_to_read -= SlotSize;
                wr_slot_idx_ = advance_slot_idx(wr_slot_idx_);
                num_used_slots_++;
            }
            // Read final slot (may be partial)
            noc_async_read(src_addr, slot_addrs_[wr_slot_idx_], bytes_to_read);
            wr_slot_idx_ = advance_slot_idx(wr_slot_idx_);
            num_used_slots_++;

            noc_async_read_barrier();

            // Update response: advance read address, update byte counters
            response->complete_allocation(bytes_allocated);
        }
    }

    // Deallocate num_slots from front
    FORCE_INLINE void cb_deallocate_slots(uint32_t num_slots) {
        rd_slot_idx_ = advance_slot_idx_n(rd_slot_idx_, num_slots);
        num_used_slots_ -= num_slots;
    }
};

}  // namespace tt::tt_fabric::udm
