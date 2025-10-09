// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tracy_memory_monitor.hpp"
#include <iostream>
#include <vector>

#ifdef TRACY_ENABLE
#include "tracy/Tracy.hpp"
#endif

namespace tt::tt_metal {

const char* TracyMemoryMonitor::get_tracy_pool_name(BufferType type, int device_id) {
    // Create static pool names for each combination
    // Tracy requires string literals, so we use static strings
    static thread_local char pool_name[64];

    const char* type_str;
    switch (type) {
        case BufferType::DRAM: type_str = "DRAM"; break;
        case BufferType::L1: type_str = "L1"; break;
        case BufferType::SYSTEM_MEMORY: type_str = "SYS_MEM"; break;
        case BufferType::L1_SMALL: type_str = "L1_SMALL"; break;
        case BufferType::TRACE: type_str = "TRACE"; break;
        default: type_str = "UNKNOWN"; break;
    }

    snprintf(pool_name, sizeof(pool_name), "TT_Dev%d_%s", device_id, type_str);
    return pool_name;
}

void TracyMemoryMonitor::track_allocation(int device_id, uint64_t buffer_id, uint64_t size, BufferType buffer_type) {
    if (device_id < 0 || device_id >= MAX_DEVICES) {
        // std::cerr << "TracyMemoryMonitor: Invalid device_id " << device_id << " in track_allocation" << std::endl;
        return;
    }

    // Update atomic counters (lock-free)
    auto& stats = device_stats_[device_id];

    switch (buffer_type) {
        case BufferType::DRAM: stats.dram_allocated.fetch_add(size, std::memory_order_relaxed); break;
        case BufferType::L1: stats.l1_allocated.fetch_add(size, std::memory_order_relaxed); break;
        case BufferType::SYSTEM_MEMORY: stats.system_memory_allocated.fetch_add(size, std::memory_order_relaxed); break;
        case BufferType::L1_SMALL: stats.l1_small_allocated.fetch_add(size, std::memory_order_relaxed); break;
        case BufferType::TRACE: stats.trace_allocated.fetch_add(size, std::memory_order_relaxed); break;
    }

    stats.num_buffers.fetch_add(1, std::memory_order_relaxed);
    stats.total_allocs.fetch_add(1, std::memory_order_relaxed);

    // Record buffer info (requires lock)
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);

        BufferKey key{device_id, buffer_id};
        BufferInfo info{buffer_id, device_id, size, buffer_type, std::chrono::steady_clock::now()};

        // Check for double allocation (potential bug)
        if (active_buffers_.find(key) != active_buffers_.end()) {
            // std::cerr << "TracyMemoryMonitor: Double allocation detected: device=" << device_id
            //  << ", buffer_id=0x" << std::hex << buffer_id << std::dec
            //  << ", size=" << size << std::endl;
        }

        active_buffers_[key] = info;
    }

    // Report to Tracy profiler (if enabled)
#ifdef TRACY_ENABLE
    // Use buffer_id as the memory address for Tracy
    // Tracy tracks memory by address, so this gives us per-buffer visibility
    const char* pool_name = get_tracy_pool_name(buffer_type, device_id);
    TracyAllocN(reinterpret_cast<void*>(buffer_id), size, pool_name);
#endif
}

void TracyMemoryMonitor::track_deallocation(int device_id, uint64_t buffer_id) {
    if (device_id < 0 || device_id >= MAX_DEVICES) {
        // std::cerr << "TracyMemoryMonitor: Invalid device_id " << device_id << " in track_deallocation" << std::endl;
        return;
    }

    BufferKey key{device_id, buffer_id};
    BufferInfo info;
    bool found = false;

    // Retrieve and remove buffer info (requires lock)
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        auto it = active_buffers_.find(key);

        if (it != active_buffers_.end()) {
            info = it->second;
            found = true;
            active_buffers_.erase(it);
        }
    }

    if (!found) {
        // std::cerr << "TracyMemoryMonitor: Deallocation of unknown buffer: device=" << device_id
        //  << ", buffer_id=0x" << std::hex << buffer_id << std::dec << std::endl;
        return;
    }

    // Update atomic counters (lock-free)
    auto& stats = device_stats_[device_id];

    switch (info.buffer_type) {
        case BufferType::DRAM: stats.dram_allocated.fetch_sub(info.size, std::memory_order_relaxed); break;
        case BufferType::L1: stats.l1_allocated.fetch_sub(info.size, std::memory_order_relaxed); break;
        case BufferType::SYSTEM_MEMORY:
            stats.system_memory_allocated.fetch_sub(info.size, std::memory_order_relaxed);
            break;
        case BufferType::L1_SMALL: stats.l1_small_allocated.fetch_sub(info.size, std::memory_order_relaxed); break;
        case BufferType::TRACE: stats.trace_allocated.fetch_sub(info.size, std::memory_order_relaxed); break;
    }

    stats.num_buffers.fetch_sub(1, std::memory_order_relaxed);
    stats.total_frees.fetch_add(1, std::memory_order_relaxed);

    // Report to Tracy profiler (if enabled)
#ifdef TRACY_ENABLE
    const char* pool_name = get_tracy_pool_name(info.buffer_type, device_id);
    TracyFreeN(reinterpret_cast<void*>(buffer_id), pool_name);
#endif
}

size_t TracyMemoryMonitor::get_active_buffer_count(int device_id) const {
    if (device_id < 0 || device_id >= MAX_DEVICES) {
        return 0;
    }

    std::unique_lock<std::mutex> lock(registry_mutex_);

    size_t count = 0;
    for (const auto& [key, info] : active_buffers_) {
        if (key.device_id == device_id) {
            count++;
        }
    }

    return count;
}

std::vector<TracyMemoryMonitor::BufferInfo> TracyMemoryMonitor::get_active_buffers(int device_id) const {
    std::vector<BufferInfo> result;

    if (device_id < 0 || device_id >= MAX_DEVICES) {
        return result;
    }

    std::unique_lock<std::mutex> lock(registry_mutex_);

    for (const auto& [key, info] : active_buffers_) {
        if (key.device_id == device_id) {
            result.push_back(info);
        }
    }

    return result;
}

void TracyMemoryMonitor::reset() {
    // Reset all atomic counters
    for (auto& stats : device_stats_) {
        stats.dram_allocated.store(0, std::memory_order_relaxed);
        stats.l1_allocated.store(0, std::memory_order_relaxed);
        stats.system_memory_allocated.store(0, std::memory_order_relaxed);
        stats.l1_small_allocated.store(0, std::memory_order_relaxed);
        stats.trace_allocated.store(0, std::memory_order_relaxed);
        stats.num_buffers.store(0, std::memory_order_relaxed);
        stats.total_allocs.store(0, std::memory_order_relaxed);
        stats.total_frees.store(0, std::memory_order_relaxed);
    }

    // Clear active buffers
    std::unique_lock<std::mutex> lock(registry_mutex_);
    active_buffers_.clear();
}

}  // namespace tt::tt_metal
