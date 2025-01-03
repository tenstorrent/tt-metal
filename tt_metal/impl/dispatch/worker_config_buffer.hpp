// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>
#include "llrt/hal.hpp"

namespace tt {

namespace tt_metal {

struct ConfigBufferEntry {
    uint32_t addr;
    uint32_t size;
    uint32_t sync_count;
};

struct ConfigBufferSync {
    bool need_sync;
    uint32_t sync_count;
};

// Manages the kernel configuration buffer on the device
// Interface is stateful, use in the sequence below
//
// Usage:
//   construct with vectors of the kernel config buffer base address and size for each managed CoreType
//   call reserve to reserve space for each CoreType, returns true if sync is required (along w/ sync values)
//   issue a sync for each CoreType that needs a sync
//   call free to free the memory just synced (mandatory whenever a client does a sync)
//   call alloc with the new sync values to allocate the memory for each CoreType
//
class WorkerConfigBufferMgr {
public:
    WorkerConfigBufferMgr();

    void init_add_buffer(uint32_t base_addr, uint32_t size);
    const std::pair<ConfigBufferSync, std::vector<ConfigBufferEntry>&> reserve(const std::vector<uint32_t>& sizes);
    void free(uint32_t free_up_to_sync_count);
    void alloc(uint32_t when_freeable_sync_count);
    void mark_completely_full(uint32_t sync);

    // Test/Debug
    uint32_t get_last_slot_addr(HalProgrammableCoreType programmable_core_type) const;

    void PrintStatus();

private:
    std::vector<uint32_t> base_addrs_;
    std::vector<uint32_t> end_addrs_;
    std::vector<std::vector<ConfigBufferEntry>> entries_;  // ring buffer of allocated space
    std::vector<uint32_t> alloc_index_;                    // always points to a valid entry
    std::vector<uint32_t> free_index_;                     // points to the next entry to free

    std::vector<ConfigBufferEntry> reservation_;
};

}  // namespace tt_metal
}  // namespace tt
