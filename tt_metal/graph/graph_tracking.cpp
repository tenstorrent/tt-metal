// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <graph_tracking.hpp>

#include <tt_stl/assert.hpp>
#include "tt_metal/impl/allocator/allocation_client.hpp"
#include "tt_metal/impl/profiler/tracy_memory_monitor.hpp"
#include <tt-metalium/mesh_device.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <mutex>
#include <unordered_map>

namespace tt {
namespace tt_metal {
class Buffer;
class IDevice;
namespace distributed {
class MeshDevice;
}
}  // namespace tt_metal
}  // namespace tt

// Global tracking for L1 allocation statistics
namespace {

// CRITICAL: Global mutex to serialize all buffer tracking calls
// This prevents race conditions where concurrent allocations/deallocations
// at the same address send out-of-order messages to the tracking server
std::mutex g_allocation_tracking_mutex;

struct L1Stats {
    std::mutex mutex;
    std::unordered_map<int, uint64_t> device_l1_allocated;  // Total L1 per device
    std::unordered_map<int, uint64_t> device_l1_peak;       // Peak L1 per device
    std::unordered_map<int, int> device_alloc_count;        // Number of L1 buffers per device
    std::chrono::steady_clock::time_point start_time;
    bool initialized = false;

    void init() {
        if (!initialized) {
            start_time = std::chrono::steady_clock::now();
            initialized = true;
        }
    }

    double elapsed_ms() {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_time).count();
    }

    void track_alloc(int device_id, uint64_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        init();
        device_l1_allocated[device_id] += size;
        device_alloc_count[device_id]++;
        if (device_l1_allocated[device_id] > device_l1_peak[device_id]) {
            device_l1_peak[device_id] = device_l1_allocated[device_id];
            // Log when we hit a new peak > 1MB
            if (device_l1_peak[device_id] > 1024 * 1024) {
                std::cout << "[" << std::fixed << std::setprecision(0) << elapsed_ms() << "ms] L1 Peak on device "
                          << device_id << ": " << (device_l1_peak[device_id] / (1024.0 * 1024.0)) << " MB ("
                          << device_alloc_count[device_id] << " buffers)" << std::endl;
            }
        }
    }

    void track_dealloc(int device_id, uint64_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        init();
        if (device_l1_allocated[device_id] >= size) {
            uint64_t before = device_l1_allocated[device_id];
            device_l1_allocated[device_id] -= size;
            device_alloc_count[device_id]--;
            // Log significant drops (> 10MB freed)
            if (before > 10 * 1024 * 1024 && device_l1_allocated[device_id] < 1024 * 1024) {
                std::cout << "[" << std::fixed << std::setprecision(0) << elapsed_ms() << "ms] L1 Drop on device "
                          << device_id << ": " << (before / (1024.0 * 1024.0)) << " MB -> "
                          << (device_l1_allocated[device_id] / (1024.0 * 1024.0)) << " MB" << std::endl;
            }
        }
    }

    void print_summary() {
        std::lock_guard<std::mutex> lock(mutex);
        std::cout << "\n=== L1 Memory Summary ===" << std::endl;
        for (int i = 0; i < 8; i++) {
            if (device_l1_peak.count(i)) {
                std::cout << "Device " << i << ": Peak " << (device_l1_peak[i] / (1024.0 * 1024.0)) << " MB, "
                          << "Current " << (device_l1_allocated[i] / (1024.0 * 1024.0)) << " MB" << std::endl;
            }
        }
        std::cout << "========================\n" << std::endl;
    }
};

L1Stats& get_l1_stats() {
    static L1Stats stats;
    return stats;
}
}  // anonymous namespace

namespace tt::tt_metal {

bool GraphTracker::is_enabled() const { return (not processors.empty()); }

void GraphTracker::push_processor(const std::shared_ptr<IGraphProcessor>& new_processor) {
    processors.push_back(new_processor);
}

void GraphTracker::pop_processor() {
    TT_ASSERT(not processors.empty(), "No processor to pop");
    processors.pop_back();
}

bool GraphTracker::add_hook(const std::shared_ptr<IGraphHooks>& new_hook) {
    if (hook) {
        return false;
    }
    hook = new_hook;
    return true;
}

void GraphTracker::track_allocate(const Buffer* buffer) {
    // Report to allocation tracking server (catches ALL allocations, hooked or not)
    // Skip reporting if the buffer's device is a MeshDevice (backing buffer)
    // The device-local buffers will be reported instead
    if (buffer->device() != nullptr) {
        // Check if this is a MeshDevice (backing buffer) - don't report these
        if (dynamic_cast<const distributed::MeshDevice*>(buffer->device()) != nullptr) {
            return;  // Skip tracking for backing buffers on MeshDevice
        }

        // Track L1 allocations for debugging
        if (buffer->buffer_type() == BufferType::L1) {
            get_l1_stats().track_alloc(buffer->device()->id(), buffer->size());
        }

        // CRITICAL: Serialize tracking calls to prevent race conditions
        // Without this mutex, concurrent allocations/deallocations at the same address
        // can send out-of-order messages to the tracking server, causing "unknown buffer" warnings
        std::lock_guard<std::mutex> tracking_lock(g_allocation_tracking_mutex);

        // Report to legacy allocation server (if enabled)
        if (AllocationClient::is_enabled()) {
            AllocationClient::report_allocation(
                buffer->device()->id(), buffer->size(), static_cast<uint8_t>(buffer->buffer_type()), buffer->address());
        }

        // Report to Tracy-based memory monitor (always enabled, checks Tracy at runtime)
        TracyMemoryMonitor::instance().track_allocation(
            buffer->device()->id(),
            buffer->address(),
            buffer->size(),
            static_cast<TracyMemoryMonitor::BufferType>(buffer->buffer_type()));
    }

    // Original graph tracking
    if (processors.empty()) {
        return;
    }
    for (auto& it : processors) {
        it->track_allocate(buffer);
    }
}

void GraphTracker::track_deallocate(Buffer* buffer) {
    // Report to allocation tracking server (catches ALL deallocations, hooked or not)
    // Skip reporting if the buffer's device is a MeshDevice (backing buffer)
    // The device-local buffers will be reported instead
    if (buffer->device() != nullptr) {
        // Check if this is a MeshDevice (backing buffer) - don't report these
        if (dynamic_cast<const distributed::MeshDevice*>(buffer->device()) != nullptr) {
            return;  // Skip tracking for backing buffers on MeshDevice
        }

        // Track L1 deallocations for debugging
        if (buffer->buffer_type() == BufferType::L1) {
            get_l1_stats().track_dealloc(buffer->device()->id(), buffer->size());
        }

        // CRITICAL: Serialize tracking calls to prevent race conditions
        // Use the same mutex as track_allocate() to ensure ordering
        std::lock_guard<std::mutex> tracking_lock(g_allocation_tracking_mutex);

        // Report to legacy allocation server (if enabled)
        if (AllocationClient::is_enabled()) {
            AllocationClient::report_deallocation(buffer->device()->id(), buffer->address());
        }

        // Report to Tracy-based memory monitor (always enabled, checks Tracy at runtime)
        TracyMemoryMonitor::instance().track_deallocation(buffer->device()->id(), buffer->address());
    }

    // Original graph tracking
    if (processors.empty()) {
        return;
    }
    for (auto& it : processors) {
        it->track_deallocate(buffer);
    }
}

void GraphTracker::track_allocate_cb(
    const CoreRangeSet& core_range_set,
    uint64_t addr,
    uint64_t size,
    bool is_globally_allocated,
    const IDevice* device) {
    // Store CB allocation for later deallocation
    {
        std::lock_guard<std::mutex> lock(cb_mutex);
        device_cb_allocations[device].push_back({addr, size});
    }

    // Report circular buffer allocation to tracking server
    if (device != nullptr) {
        // CRITICAL: Serialize tracking calls to prevent race conditions
        std::lock_guard<std::mutex> tracking_lock(g_allocation_tracking_mutex);

        // Report to legacy allocation server (if enabled)
        if (AllocationClient::is_enabled()) {
            // Circular buffers are always L1
            AllocationClient::report_allocation(device->id(), size, static_cast<uint8_t>(BufferType::L1), addr);
        }

        // Report to Tracy-based memory monitor (circular buffers are L1)
        TracyMemoryMonitor::instance().track_allocation(device->id(), addr, size, TracyMemoryMonitor::BufferType::L1);
    }

    // Original graph tracking
    if (processors.empty()) {
        return;
    }
    for (auto& it : processors) {
        it->track_allocate_cb(core_range_set, addr, size, is_globally_allocated, device);
    }
}

void GraphTracker::track_deallocate_cb(const IDevice* device) {
    // Report all CB deallocations for this device using stored addresses
    std::vector<CBAllocation> cbs_to_deallocate;
    {
        std::lock_guard<std::mutex> lock(cb_mutex);
        auto it = device_cb_allocations.find(device);
        if (it != device_cb_allocations.end()) {
            cbs_to_deallocate = std::move(it->second);
            device_cb_allocations.erase(it);
        }
    }

    // Report each CB deallocation to the tracking server
    if (device != nullptr) {
        // CRITICAL: Serialize tracking calls to prevent race conditions
        std::lock_guard<std::mutex> tracking_lock(g_allocation_tracking_mutex);

        for (const auto& cb : cbs_to_deallocate) {
            // Report to legacy allocation server (if enabled)
            if (AllocationClient::is_enabled()) {
                AllocationClient::report_deallocation(device->id(), cb.addr);
            }

            // Report to Tracy-based memory monitor
            TracyMemoryMonitor::instance().track_deallocation(device->id(), cb.addr);
        }
    }

    // Original graph tracking
    if (processors.empty()) {
        return;
    }
    for (auto& it : processors) {
        it->track_deallocate_cb(device);
    }
}

void GraphTracker::track_program(Program* program, const IDevice* device) {
    TT_ASSERT(program);
    TT_ASSERT(device);
    if (processors.empty()) {
        return;
    }
    for (auto& it : processors) {
        it->track_program(program, device);
    }
}

bool GraphTracker::hook_allocate(const Buffer* buffer) {
    if (hook == nullptr) {
        return false;
    }

    bool hooked = hook->hook_allocate(buffer);
    if (hooked) {
        std::lock_guard<std::mutex> lock(hooked_buffers_mutex);
        bool inserted = hooked_buffers.insert(buffer).second;
        TT_FATAL(inserted, "Can't hook allocation of a buffer which is already allocated");
    }
    return hooked;
}

bool GraphTracker::hook_deallocate(Buffer* buffer) {
    if (hook == nullptr) {
        return false;
    }

    bool hooked = hook->hook_deallocate(buffer);
    if (hooked) {
        std::lock_guard<std::mutex> lock(hooked_buffers_mutex);
        auto buffer_it = hooked_buffers.find(buffer);
        TT_FATAL(
            buffer_it != hooked_buffers.end(), "Can't hook deallocation of a buffer which allocation wasn't hooked");
        hooked_buffers.erase(buffer_it);
    }
    return hooked;
}

bool GraphTracker::hook_write_to_device(const tt::tt_metal::Buffer* buffer) {
    if (hook == nullptr) {
        return false;
    }
    return hook->hook_write_to_device(buffer);
}

bool GraphTracker::hook_write_to_device(const tt::tt_metal::distributed::MeshBuffer* mesh_buffer) {
    if (hook == nullptr) {
        return false;
    }
    return hook->hook_write_to_device(mesh_buffer);
}

bool GraphTracker::hook_read_from_device(tt::tt_metal::Buffer* buffer) {
    if (hook == nullptr) {
        return false;
    }
    return hook->hook_read_from_device(buffer);
}

bool GraphTracker::hook_read_from_device(const tt::tt_metal::distributed::MeshBuffer* mesh_buffer) {
    if (hook == nullptr) {
        return false;
    }
    return hook->hook_read_from_device(mesh_buffer);
}

bool GraphTracker::hook_program(tt::tt_metal::Program* program) {
    if (hook == nullptr) {
        return false;
    }
    return hook->hook_program(program);
}

const std::vector<std::shared_ptr<IGraphProcessor>>& GraphTracker::get_processors() const { return processors; }

const std::shared_ptr<IGraphHooks>& GraphTracker::get_hook() const { return hook; }

void GraphTracker::clear() {
    processors.clear();
    clear_hook();
}

void GraphTracker::print_l1_summary() { get_l1_stats().print_summary(); }

void GraphTracker::clear_hook() {
    hooked_buffers.clear();
    hook = nullptr;
}

}  // namespace tt::tt_metal
