// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/memory_tracking/shm_tracking_processor.hpp"
#include "impl/memory_tracking/memory_stats_shm.hpp"
#include "impl/context/metal_context.hpp"
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

ShmTrackingProcessor::ShmTrackingProcessor() :
    verbose_enabled_(MetalContext::instance().rtoptions().get_shm_verbose()) {}

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
        // MeshBuffer::create() allocates a backing buffer on the MeshDevice with
        // device_local_size (the per-device portion for sharded, or full size for
        // replicated). Per-device buffers created by initialize_device_buffers() use
        // the address-taking Buffer::create overload (owns_data_=false) and do NOT
        // trigger track_allocate, so there is no double-counting.
        // buffer->size() is already the correct per-device size in both layouts.
        std::lock_guard<std::mutex> tracking_lock(tracking_mutex_);

        auto underlying_devices = mesh_device->get_devices();
        size_t num_tracked = 0;
        uint64_t size_per_device = buffer->size();

        // Track on all underlying Devices that have SHM provider
        for (auto* underlying_device : underlying_devices) {
            const auto* device = dynamic_cast<const Device*>(underlying_device);
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
                            static_cast<unsigned>(buffer->buffer_type()),
                            size_per_device,
                            getpid());
                    }
                    shm_provider->record_allocation(
                        getpid(),
                        size_per_device,
                        to_shm_buffer_type(buffer->buffer_type()),
                        static_cast<uint32_t>(device->id()));
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
                    static_cast<unsigned>(buffer->buffer_type()),
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

    const auto* device = dynamic_cast<const Device*>(buffer->device());
    if (device) {
        auto* shm_provider = device->get_shm_stats_provider();
        if (shm_provider) {
            if (verbose_enabled_) {
                log_debug(
                    tt::LogMetal,
                    "ShmTrackingProcessor::track_allocate: Calling SHM record_allocation (device_id={}, type={}, "
                    "size={} B, pid={})",
                    buffer->device()->id(),
                    static_cast<unsigned>(buffer->buffer_type()),
                    buffer->size(),
                    getpid());
            }
            shm_provider->record_allocation(
                getpid(),
                buffer->size(),
                to_shm_buffer_type(buffer->buffer_type()),
                static_cast<uint32_t>(buffer->device()->id()));
        } else {
            // SHM provider not available (might be disabled or not initialized)
            if (verbose_enabled_) {
                log_warning(
                    tt::LogMetal,
                    "ShmTrackingProcessor::track_allocate: SHM provider is NULL for device {} (buffer type: {}, "
                    "size: {} B)",
                    buffer->device()->id(),
                    static_cast<unsigned>(buffer->buffer_type()),
                    buffer->size());
            } else {
                static bool warned = false;
                if (!warned) {
                    log_warning(
                        tt::LogMetal,
                        "SHM provider not available for device {} (buffer type: {})",
                        buffer->device()->id(),
                        static_cast<unsigned>(buffer->buffer_type()));
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
                static_cast<unsigned>(buffer->buffer_type()),
                buffer->size(),
                static_cast<const void*>(buffer->device()));
        } else {
            static bool warned_cast = false;
            if (!warned_cast) {
                log_warning(
                    tt::LogMetal,
                    "Failed to cast buffer->device() to Device* for buffer type {}",
                    static_cast<unsigned>(buffer->buffer_type()));
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
        // Mirror track_allocate: buffer->size() is device_local_size (correct for both
        // sharded and replicated). No double-counting — see track_allocate comment.
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
                        getpid(),
                        size_per_device,
                        to_shm_buffer_type(buffer->buffer_type()),
                        static_cast<uint32_t>(device->id()));
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
                getpid(),
                buffer->size(),
                to_shm_buffer_type(buffer->buffer_type()),
                static_cast<uint32_t>(buffer->device()->id()));
        }
    }
}

}  // namespace tt::tt_metal
