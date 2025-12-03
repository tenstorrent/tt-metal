// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "debug/assert.h"
#include "dataflow_api.h"
#include "noc_parameters.h"

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

    // Check if num_slots are available
    FORCE_INLINE bool cb_has_enough_slots(uint32_t num_slots) const {
        return (NumSlots - num_used_slots_) >= num_slots;
    }

    // Helper: calculate slots needed for a given size
    FORCE_INLINE uint32_t slots_needed(uint32_t size_bytes) const { return (size_bytes + SlotSize - 1) / SlotSize; }

    // Get slot size
    FORCE_INLINE constexpr uint32_t get_slot_size() const { return SlotSize; }

    // Get slot address by index
    FORCE_INLINE uint32_t get_slot_addr(uint32_t idx) const { return slot_addrs_[idx]; }

    // Get current read slot index
    FORCE_INLINE uint32_t get_rd_slot_idx() const { return rd_slot_idx_; }

    // Get next slot index (with wrap)
    FORCE_INLINE uint32_t get_next_slot_idx(uint32_t idx) const { return advance_slot_idx(idx); }

    // Allocate slots and fill with data from noc_addr using noc_async_read
    FORCE_INLINE void cb_allocate_and_fill_slots(uint64_t noc_addr, uint32_t size_bytes) {
        uint32_t bytes_remaining = size_bytes;
        uint64_t src_addr = noc_addr;

        // Read slot by slot
        while (bytes_remaining > 0) {
            uint32_t read_size = (bytes_remaining > SlotSize) ? SlotSize : bytes_remaining;
            noc_async_read(src_addr, slot_addrs_[wr_slot_idx_], read_size);

            src_addr += read_size;
            bytes_remaining -= read_size;
            wr_slot_idx_ = advance_slot_idx(wr_slot_idx_);
            num_used_slots_++;
        }
    }

    // Deallocate num_slots from front
    FORCE_INLINE void cb_deallocate_slots(uint32_t num_slots) {
        rd_slot_idx_ = advance_slot_idx_n(rd_slot_idx_, num_slots);
        num_used_slots_ -= num_slots;
    }

    FORCE_INLINE void reset() {
        wr_slot_idx_ = 0;
        rd_slot_idx_ = 0;
        num_used_slots_ = 0;
    }
};

}  // namespace tt::tt_fabric::udm
