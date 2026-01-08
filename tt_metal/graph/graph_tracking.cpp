// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <graph_tracking.hpp>

#include <tt_stl/assert.hpp>
#include "impl/profiler/memory_stats_shm.hpp"
#include "impl/device/device_impl.hpp"
#include "buffer.hpp"
#include <tt-metalium/mesh_device.hpp>
#include <tt-logger/tt-logger.hpp>
#include <unistd.h>  // for getpid()
#include <mutex>

namespace tt::tt_metal {
class Buffer;
class IDevice;
namespace distributed {
class MeshDevice;
}
}  // namespace tt::tt_metal

// Global tracking for L1 allocation statistics
namespace {

// CRITICAL: Global mutex to serialize all buffer tracking calls
// This prevents race conditions where concurrent allocations/deallocations
// at the same address send out-of-order messages to the tracking server
std::mutex g_allocation_tracking_mutex;

// Helper: Convert BufferType to ShmBufferType for shared memory tracking
tt::tt_metal::ShmBufferType to_shm_buffer_type(tt::tt_metal::BufferType type) {
    using namespace tt::tt_metal;
    switch (type) {
        case BufferType::DRAM: return ShmBufferType::DRAM;
        case BufferType::L1: return ShmBufferType::L1;
        case BufferType::L1_SMALL: return ShmBufferType::L1_SMALL;
        case BufferType::TRACE: return ShmBufferType::TRACE;
        case BufferType::SYSTEM_MEMORY: return ShmBufferType::SYSTEM_MEMORY;
        default: return ShmBufferType::UNKNOWN;
    }
}

}  // anonymous namespace

namespace tt::tt_metal {

GraphTracker& GraphTracker::instance() {
    static GraphTracker tracker;
    return tracker;
}

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
    // Optional verbose logging for SHM tracking
    static bool verbose_enabled = [] {
        const char* env = std::getenv("TT_METAL_SHM_VERBOSE");
        return env && std::string(env) == "1";
    }();

    // Report to shared memory tracking (for tt-smi-ui)
    // For MeshDevice backing buffers, track on underlying Device objects
    if (buffer->device() != nullptr) {
        // Check if this is a MeshDevice (backing buffer)
        const auto* mesh_device = dynamic_cast<const distributed::MeshDevice*>(buffer->device());
        if (mesh_device != nullptr) {
            // For MeshDevice backing buffers, track on ALL underlying Device objects
            // Note: This may cause some over-counting if buffers are replicated, but ensures
            // all devices are tracked. For sharded buffers, each device gets its portion.
            std::lock_guard<std::mutex> tracking_lock(g_allocation_tracking_mutex);

            auto underlying_devices = mesh_device->get_devices();
            size_t num_tracked = 0;

            // Calculate size per device (assuming equal distribution for sharded buffers)
            // For replicated buffers, this will cause over-counting, but it's better than missing data
            uint64_t size_per_device = buffer->size();
            // Option: divide by device count for sharded buffers, but we don't know if it's sharded or replicated
            // For now, track full size on each device - user can see per-device usage

            // Track on all underlying Devices that have SHM provider
            for (auto* underlying_device : underlying_devices) {
                auto* device = dynamic_cast<const Device*>(underlying_device);
                if (device) {
                    auto* shm_provider = device->get_shm_stats_provider();
                    if (shm_provider) {
                        // Track allocation on this underlying device
                        if (verbose_enabled) {
                            log_debug(
                                tt::LogMetal,
                                "GraphTracker::track_allocate: Tracking MeshDevice backing buffer on underlying device "
                                "{} "
                                "(mesh_device_id={}, type={}, size={} B, pid={})",
                                device->id(),
                                buffer->device()->id(),
                                static_cast<int>(buffer->buffer_type()),
                                size_per_device,
                                getpid());
                        }
                        shm_provider->record_allocation(
                            getpid(), size_per_device, to_shm_buffer_type(buffer->buffer_type()), device->id());
                        num_tracked++;
                    }
                }
            }

            if (verbose_enabled) {
                if (num_tracked == 0) {
                    log_warning(
                        tt::LogMetal,
                        "GraphTracker::track_allocate: MeshDevice backing buffer could not be tracked on any "
                        "underlying device "
                        "(mesh_device_id={}, num_devices={}, type={}, size={} B)",
                        buffer->device()->id(),
                        underlying_devices.size(),
                        static_cast<int>(buffer->buffer_type()),
                        buffer->size());
                } else {
                    log_debug(
                        tt::LogMetal,
                        "GraphTracker::track_allocate: Tracked MeshDevice backing buffer on {} underlying device(s) "
                        "(mesh_device_id={}, total_devices={})",
                        num_tracked,
                        buffer->device()->id(),
                        underlying_devices.size());
                }
            }

            // Continue to original graph tracking
            if (processors.empty()) {
                return;
            }
            for (auto& it : processors) {
                it->track_allocate(buffer);
            }
            return;
        }

        // CRITICAL: Serialize tracking calls to prevent race conditions
        // Without this mutex, concurrent allocations/deallocations at the same address
        // can send out-of-order messages, causing tracking inconsistencies
        std::lock_guard<std::mutex> tracking_lock(g_allocation_tracking_mutex);

        // Report to shared memory tracking (if enabled)
        // Use cumulative tracking for all buffer allocations
        auto* device = dynamic_cast<const Device*>(buffer->device());
        if (device) {
            auto* shm_provider = device->get_shm_stats_provider();
            if (shm_provider) {
                if (verbose_enabled) {
                    log_debug(
                        tt::LogMetal,
                        "GraphTracker::track_allocate: Calling SHM record_allocation (device_id={}, type={}, size={} "
                        "B, pid={})",
                        buffer->device()->id(),
                        static_cast<int>(buffer->buffer_type()),
                        buffer->size(),
                        getpid());
                }
                shm_provider->record_allocation(
                    getpid(), buffer->size(), to_shm_buffer_type(buffer->buffer_type()), buffer->device()->id());
            } else {
                // Debug: SHM provider not available (might be disabled or not initialized)
                if (verbose_enabled) {
                    log_warning(
                        tt::LogMetal,
                        "GraphTracker::track_allocate: SHM provider is NULL for device {} (buffer type: {}, size: {} "
                        "B)",
                        buffer->device()->id(),
                        static_cast<int>(buffer->buffer_type()),
                        buffer->size());
                } else {
                    static bool warned = false;
                    if (!warned) {
                        log_warning(
                            tt::LogMetal,
                            "SHM provider not available for device {} (buffer type: {})",
                            buffer->device()->id(),
                            static_cast<int>(buffer->buffer_type()));
                        warned = true;  // Only warn once to avoid spam
                    }
                }
            }
        } else {
            // Debug: Failed to cast buffer->device() to Device*
            if (verbose_enabled) {
                log_warning(
                    tt::LogMetal,
                    "GraphTracker::track_allocate: dynamic_cast<Device*> failed (device_id={}, buffer_type={}, size={} "
                    "B, device_ptr={})",
                    buffer->device()->id(),
                    static_cast<int>(buffer->buffer_type()),
                    buffer->size(),
                    static_cast<const void*>(buffer->device()));
            } else {
                static bool warned_cast = false;
                if (!warned_cast) {
                    log_warning(
                        tt::LogMetal,
                        "Failed to cast buffer->device() to Device* for buffer type {}",
                        static_cast<int>(buffer->buffer_type()));
                    warned_cast = true;
                }
            }
        }
    } else {
        if (verbose_enabled) {
            log_warning(
                tt::LogMetal,
                "GraphTracker::track_allocate: buffer->device() is nullptr (type={}, size={} B)",
                static_cast<int>(buffer->buffer_type()),
                buffer->size());
        }
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
    // Report to shared memory tracking (for tt-smi-ui)
    // For MeshDevice backing buffers, track on underlying Device objects
    if (buffer->device() != nullptr) {
        // Check if this is a MeshDevice (backing buffer)
        const auto* mesh_device = dynamic_cast<const distributed::MeshDevice*>(buffer->device());
        if (mesh_device != nullptr) {
            // For MeshDevice backing buffers, track on ALL underlying Device objects
            std::lock_guard<std::mutex> tracking_lock(g_allocation_tracking_mutex);

            auto underlying_devices = mesh_device->get_devices();
            uint64_t size_per_device = buffer->size();

            // Track deallocation on all underlying Devices that have SHM provider
            for (auto* underlying_device : underlying_devices) {
                auto* device = dynamic_cast<Device*>(underlying_device);
                if (device) {
                    auto* shm_provider = device->get_shm_stats_provider();
                    if (shm_provider) {
                        // Track deallocation on this underlying device
                        shm_provider->record_deallocation(
                            getpid(), size_per_device, to_shm_buffer_type(buffer->buffer_type()), device->id());
                    }
                }
            }

            // Continue to original graph tracking
            if (processors.empty()) {
                return;
            }
            for (auto& it : processors) {
                it->track_deallocate(buffer);
            }
            return;
        }

        // CRITICAL: Serialize tracking calls to prevent race conditions
        // Use the same mutex as track_allocate() to ensure ordering
        std::lock_guard<std::mutex> tracking_lock(g_allocation_tracking_mutex);

        // Report to shared memory tracking (if enabled)
        // Use cumulative tracking for all buffer deallocations
        auto* device = dynamic_cast<Device*>(buffer->device());
        if (device) {
            auto* shm_provider = device->get_shm_stats_provider();
            if (shm_provider) {
                shm_provider->record_deallocation(
                    getpid(), buffer->size(), to_shm_buffer_type(buffer->buffer_type()), buffer->device()->id());
            }
        }
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
    if (processors.empty()) {
        return;
    }
    for (auto& it : processors) {
        it->track_allocate_cb(core_range_set, addr, size, is_globally_allocated, device);
    }
}

void GraphTracker::track_deallocate_cb(const IDevice* device) {
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
        if (buffer_it == hooked_buffers.end()) {
            log_warning(tt::LogMetal, "Can't hook deallocation of a buffer which allocation wasn't hooked");
        } else {
            hooked_buffers.erase(buffer_it);
        }
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

void GraphTracker::clear_hook() {
    hooked_buffers.clear();
    hook = nullptr;
}

}  // namespace tt::tt_metal
