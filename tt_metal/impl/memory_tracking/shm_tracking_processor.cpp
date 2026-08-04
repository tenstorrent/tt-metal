// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/memory_tracking/shm_tracking_processor.hpp"
#include "impl/memory_tracking/memory_stats_shm.hpp"
#include "impl/device/device_impl.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-logger/tt-logger.hpp>
#include <unistd.h>
#include <sys/syscall.h>
#include <mutex>

namespace tt::tt_metal {

// DIAGNOSTIC (temporary): OS thread id of the caller. Used to compare the thread that
// performs buffer allocations against the thread that initialized SHM / would have
// registered the (thread_local) GraphTracker processor. Remove once confirmed.
static long shm_diag_gettid() { return static_cast<long>(::syscall(SYS_gettid)); }

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

// Process-wide serialization of SHM buffer tracking. Previously the per-instance
// ShmTrackingProcessor::tracking_mutex_; now shared by the direct recording path so
// concurrent allocations/deallocations at the same address cannot send out-of-order
// updates to the SHM region.
static std::mutex& shm_tracking_mutex() {
    static std::mutex mtx;
    return mtx;
}

void shm_record_buffer_allocation(const Buffer* buffer) {
    if (!buffer || !buffer->device()) {
        return;
    }

    // DIAGNOSTIC (temporary): log the first allocation seen on each OS thread. Compare
    // these tids to the "[SHM-DIAG] init/registration thread" line from Device::initialize.
    // If they differ, the pre-fix thread_local GraphTracker processor (pushed only on the
    // init thread) never observed these allocations -- which is why the direct call fixes it.
    thread_local bool shm_diag_logged_alloc_tid = false;
    if (!shm_diag_logged_alloc_tid) {
        shm_diag_logged_alloc_tid = true;
        log_info(
            tt::LogMetal,
            "[SHM-DIAG] first buffer allocation on tid={} (buffer_type={})",
            shm_diag_gettid(),
            static_cast<unsigned>(buffer->buffer_type()));
    }

    // Check if this is a MeshDevice (backing buffer)
    const auto* mesh_device = dynamic_cast<const distributed::MeshDevice*>(buffer->device());
    if (mesh_device != nullptr) {
        // MeshBuffer::create() allocates a backing buffer on the MeshDevice with
        // device_local_size (the per-device portion for sharded, or full size for
        // replicated). Per-device buffers created by initialize_device_buffers() use
        // the address-taking Buffer::create overload (owns_data_=false) and do NOT
        // trigger allocate_impl(), so there is no double-counting.
        // buffer->size() is already the correct per-device size in both layouts.
        std::lock_guard<std::mutex> tracking_lock(shm_tracking_mutex());

        uint64_t size_per_device = buffer->size();

        // Track on all underlying Devices that have SHM provider
        for (auto* underlying_device : mesh_device->get_devices()) {
            const auto* device = dynamic_cast<const Device*>(underlying_device);
            if (device) {
                auto* shm_provider = device->get_shm_stats_provider();
                if (shm_provider) {
                    shm_provider->record_allocation(
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
    std::lock_guard<std::mutex> tracking_lock(shm_tracking_mutex());

    const auto* device = dynamic_cast<const Device*>(buffer->device());
    if (device) {
        auto* shm_provider = device->get_shm_stats_provider();
        if (shm_provider) {
            shm_provider->record_allocation(
                getpid(),
                buffer->size(),
                to_shm_buffer_type(buffer->buffer_type()),
                static_cast<uint32_t>(buffer->device()->id()));
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
}

void shm_record_buffer_deallocation(Buffer* buffer) {
    if (!buffer || !buffer->device()) {
        return;
    }

    // Check if this is a MeshDevice (backing buffer)
    const auto* mesh_device = dynamic_cast<const distributed::MeshDevice*>(buffer->device());
    if (mesh_device != nullptr) {
        // Mirror shm_record_buffer_allocation: buffer->size() is device_local_size
        // (correct for both sharded and replicated). No double-counting.
        std::lock_guard<std::mutex> tracking_lock(shm_tracking_mutex());

        uint64_t size_per_device = buffer->size();

        for (auto* underlying_device : mesh_device->get_devices()) {
            auto* device = dynamic_cast<Device*>(underlying_device);
            if (device) {
                auto* shm_provider = device->get_shm_stats_provider();
                if (shm_provider) {
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
    std::lock_guard<std::mutex> tracking_lock(shm_tracking_mutex());

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

// The processor hooks are retained for API compatibility but are no longer registered
// with GraphTracker (its processor stack is thread_local since #44668, which is the wrong
// scope for a permanent cross-thread tracker). Recording now happens via the direct
// shm_record_buffer_* entry points called from Buffer. Should the processor ever be
// re-registered, delegating keeps behaviour identical and single-counted.
ShmTrackingProcessor::ShmTrackingProcessor(bool /*verbose*/) {}

void ShmTrackingProcessor::track_allocate(const Buffer* buffer) { shm_record_buffer_allocation(buffer); }

void ShmTrackingProcessor::track_deallocate(Buffer* buffer) { shm_record_buffer_deallocation(buffer); }

}  // namespace tt::tt_metal
