// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/memory_owner.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/indestructible.hpp>

#include "impl/allocator/allocator.hpp"
#include "impl/buffers/buffer_impl.hpp"

namespace tt::tt_metal {

namespace {

// Owner names are interned once per process; id 0 is "no owner".
struct MemoryOwnerRegistry {
    std::mutex mutex;
    std::vector<std::string> names{std::string{}};
    std::unordered_map<std::string, uint32_t> ids{{std::string{}, 0}};
};

MemoryOwnerRegistry& memory_owner_registry() {
    static ttsl::Indestructible<MemoryOwnerRegistry> registry;
    return registry.get();
}

thread_local std::vector<uint32_t> memory_owner_stack;

}  // namespace

uint32_t current_memory_owner_id() { return memory_owner_stack.empty() ? 0 : memory_owner_stack.back(); }

std::string memory_owner_name(uint32_t owner_id) {
    auto& registry = memory_owner_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    TT_ASSERT(owner_id < registry.names.size(), "Unknown memory owner id {}", owner_id);
    return registry.names.at(owner_id);
}

void AllocatorImpl::accumulate_memory_usage_by_owner(MemoryUsageAccumulator& usage) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const Buffer* buffer : allocated_buffers_) {
        const auto owner_it = buffer_owners_.find(buffer);
        const uint32_t owner_id = owner_it == buffer_owners_.end() ? 0 : owner_it->second;
        auto& entry = usage[{owner_id, buffer->buffer_type()}];
        entry.num_buffers++;
        entry.total_bytes += buffer->aligned_size();
        const DeviceAddr size_per_bank = buffer->aligned_size_per_bank();
        if (buffer->impl().per_core_allocation_) {
            // Per-core buffers occupy only their own cores, each at its own address.
            for (const auto& [core, address] : buffer->impl().per_core_addresses_) {
                entry.per_core_bytes[core] += size_per_bank;
            }
        } else {
            // A lockstep buffer reserves its address range in every bank.
            entry.lockstep_bytes_per_bank += size_per_bank;
        }
    }
}

namespace experimental {

void push_memory_owner(std::string_view owner) {
    auto& registry = memory_owner_registry();
    uint32_t owner_id = 0;
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        auto [it, inserted] =
            registry.ids.try_emplace(std::string(owner), static_cast<uint32_t>(registry.names.size()));
        if (inserted) {
            registry.names.emplace_back(owner);
        }
        owner_id = it->second;
    }
    memory_owner_stack.push_back(owner_id);
}

void pop_memory_owner() {
    TT_FATAL(!memory_owner_stack.empty(), "pop_memory_owner called without a matching push_memory_owner");
    memory_owner_stack.pop_back();
}

std::string current_memory_owner() { return memory_owner_name(current_memory_owner_id()); }

std::vector<MemoryUsageByOwner> memory_usage_by_owner(const std::vector<AllocatorImpl*>& allocators) {
    AllocatorImpl::MemoryUsageAccumulator usage;
    for (const auto* allocator : allocators) {
        allocator->accumulate_memory_usage_by_owner(usage);
    }
    std::vector<MemoryUsageByOwner> result;
    result.reserve(usage.size());
    for (const auto& [key, entry] : usage) {
        DeviceAddr largest_per_core = 0;
        for (const auto& [core, bytes] : entry.per_core_bytes) {
            largest_per_core = std::max(largest_per_core, bytes);
        }
        result.push_back(MemoryUsageByOwner{
            .owner = memory_owner_name(key.first),
            .buffer_type = key.second,
            .num_buffers = entry.num_buffers,
            .total_bytes = entry.total_bytes,
            .bytes_per_bank = entry.lockstep_bytes_per_bank + largest_per_core,
        });
    }
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return std::tie(a.owner, a.buffer_type) < std::tie(b.owner, b.buffer_type);
    });
    return result;
}

}  // namespace experimental

}  // namespace tt::tt_metal
