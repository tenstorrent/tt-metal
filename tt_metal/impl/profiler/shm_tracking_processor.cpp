// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "shm_tracking_processor.hpp"
#include "memory_stats_shm.hpp"
#include "impl/device/device_impl.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-logger/tt-logger.hpp>
#include <unistd.h>

namespace tt::tt_metal {

// Convert BufferType to ShmBufferType for shared memory tracking
static ShmBufferType to_shm_buffer_type(BufferType type) {
    switch (type) {
        case BufferType::DRAM: return ShmBufferType::DRAM;
        case BufferType::L1: return ShmBufferType::L1;
        case BufferType::L1_SMALL: return ShmBufferType::L1_SMALL;
        case BufferType::TRACE: return ShmBufferType::TRACE;
        case BufferType::SYSTEM_MEMORY: return ShmBufferType::SYSTEM_MEMORY;
        default: return ShmBufferType::UNKNOWN;
    }
}

ShmTrackingProcessor::ShmTrackingProcessor() {
    // Check for verbose logging at construction time
    const char* env = std::getenv("TT_METAL_SHM_VERBOSE");
    verbose_enabled_ = (env && std::string(env) == "1");
}

void ShmTrackingProcessor::track_allocate(const Buffer* buffer) {
    if (!buffer || !buffer->device()) {
        if (verbose_enabled_) {
            log_warning(tt::LogMetal, "ShmTrackingProcessor::track_allocate: buffer or buffer->device() is nullptr");
        }
        return;
    }

    // Check if this is a MeshDevice (backing buffer)
    const auto* mesh_device = dynamic_cast<const distributed::MeshDevice*>(buffer->device());
    if (mesh_device != nullptr) {
        // For MeshDevice backing buffers, track on ALL underlying Device objects
        // This may cause some over-counting if buffers are replicated, but ensures
        // all devices are tracked. For sharded buffers, each device gets its portion.
        std::lock_guard<std::mutex> tracking_lock(tracking_mutex_);

        auto underlying_devices = mesh_device->get_devices();
        size_t num_tracked = 0;

        // Calculate size per device (assuming equal distribution for sharded buffers)
        // For replicated buffers, this will cause over-counting, but it's better than missing data
        uint64_t size_per_device = buffer->size();

        // Track on all underlying Devices that have SHM provider
        for (auto* underlying_device : underlying_devices) {
            auto* device = dynamic_cast<const Device*>(underlying_device);
            if (device) {
                auto* shm_provider = device->get_shm_stats_provider();
                if (shm_provider) {
                    // Track allocation on this underlying device
                    if (verbose_enabled_) {
                        log_debug(
                            tt::LogMetal,
                            "ShmTrackingProcessor::track_allocate: Tracking MeshDevice backing buffer on underlying "
                            "device {} (mesh_device_id={}, type={}, size={} B, pid={})",
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

        if (verbose_enabled_) {
            if (num_tracked == 0) {
                log_warning(
                    tt::LogMetal,
                    "ShmTrackingProcessor::track_allocate: MeshDevice backing buffer could not be tracked on any "
                    "underlying device (mesh_device_id={}, num_devices={}, type={}, size={} B)",
                    buffer->device()->id(),
                    underlying_devices.size(),
                    static_cast<int>(buffer->buffer_type()),
                    buffer->size());
            } else {
                log_debug(
                    tt::LogMetal,
                    "ShmTrackingProcessor::track_allocate: Tracked MeshDevice backing buffer on {} underlying "
                    "device(s) (mesh_device_id={}, total_devices={})",
                    num_tracked,
                    buffer->device()->id(),
                    underlying_devices.size());
            }
        }
        return;
    }

    // Regular Device (non-MeshDevice) tracking
    std::lock_guard<std::mutex> tracking_lock(tracking_mutex_);

    auto* device = dynamic_cast<const Device*>(buffer->device());
    if (device) {
        auto* shm_provider = device->get_shm_stats_provider();
        if (shm_provider) {
            if (verbose_enabled_) {
                log_debug(
                    tt::LogMetal,
                    "ShmTrackingProcessor::track_allocate: Calling SHM record_allocation (device_id={}, type={}, "
                    "size={} B, pid={})",
                    buffer->device()->id(),
                    static_cast<int>(buffer->buffer_type()),
                    buffer->size(),
                    getpid());
            }
            shm_provider->record_allocation(
                getpid(), buffer->size(), to_shm_buffer_type(buffer->buffer_type()), buffer->device()->id());
        } else {
            // SHM provider not available (might be disabled or not initialized)
            if (verbose_enabled_) {
                log_warning(
                    tt::LogMetal,
                    "ShmTrackingProcessor::track_allocate: SHM provider is NULL for device {} (buffer type: {}, "
                    "size: {} B)",
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
        // Failed to cast buffer->device() to Device*
        if (verbose_enabled_) {
            log_warning(
                tt::LogMetal,
                "ShmTrackingProcessor::track_allocate: dynamic_cast<Device*> failed (device_id={}, buffer_type={}, "
                "size={} B, device_ptr={})",
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
}

void ShmTrackingProcessor::track_deallocate(Buffer* buffer) {
    if (!buffer || !buffer->device()) {
        return;
    }

    // Check if this is a MeshDevice (backing buffer)
    const auto* mesh_device = dynamic_cast<const distributed::MeshDevice*>(buffer->device());
    if (mesh_device != nullptr) {
        // For MeshDevice backing buffers, track on ALL underlying Device objects
        std::lock_guard<std::mutex> tracking_lock(tracking_mutex_);

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
        return;
    }

    // Regular Device (non-MeshDevice) tracking
    std::lock_guard<std::mutex> tracking_lock(tracking_mutex_);

    auto* device = dynamic_cast<Device*>(buffer->device());
    if (device) {
        auto* shm_provider = device->get_shm_stats_provider();
        if (shm_provider) {
            shm_provider->record_deallocation(
                getpid(), buffer->size(), to_shm_buffer_type(buffer->buffer_type()), buffer->device()->id());
        }
    }
}

}  // namespace tt::tt_metal
