// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <tt-metalium/buffer_types.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::experimental {

// Experimental memory attribution for several independent workloads (for example models) sharing one device in one
// process. Device buffers allocated on this thread while an owner is set are attributed to that owner; the innermost
// owner wins and an empty name means "no owner". Attribution is recorded when a buffer is allocated and dropped when
// it is freed. With no owner set, the cost is one thread-local read per allocation.
void push_memory_owner(std::string_view owner);
void pop_memory_owner();
std::string current_memory_owner();

class MemoryOwnerGuard {
public:
    explicit MemoryOwnerGuard(std::string_view owner) { push_memory_owner(owner); }
    ~MemoryOwnerGuard() { pop_memory_owner(); }
    MemoryOwnerGuard(const MemoryOwnerGuard&) = delete;
    MemoryOwnerGuard& operator=(const MemoryOwnerGuard&) = delete;
    MemoryOwnerGuard(MemoryOwnerGuard&&) = delete;
    MemoryOwnerGuard& operator=(MemoryOwnerGuard&&) = delete;
};

struct MemoryUsageByOwner {
    std::string owner;  // empty: allocated while no owner was set
    BufferType buffer_type{};
    std::size_t num_buffers = 0;
    // Sum of the buffers' sizes over all banks.
    std::uint64_t total_bytes = 0;
    // The owner's largest footprint in any one bank: per core for L1, per channel for DRAM. A lockstep buffer
    // reserves its address range in every bank, so this is what the owner takes from every other user of the bank.
    std::uint64_t bytes_per_bank = 0;
};

// Live buffers of the mesh device's allocators (the global allocator and those of the active sub-device manager),
// one entry per owner and buffer type, sorted by owner then buffer type.
std::vector<MemoryUsageByOwner> get_memory_usage_by_owner(const distributed::MeshDevice& mesh_device);

}  // namespace tt::tt_metal::experimental
