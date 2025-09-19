// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/pinned_memory.hpp>
#include "pinned_memory_impl.hpp"

#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <memory>
#include <cstdint>
#include <unistd.h>

#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt-metalium/distributed.hpp>
#include <context/metal_context.hpp>
#include <umd/device/chip_helpers/sysmem_manager.hpp>
#include <umd/device/chip_helpers/sysmem_buffer.hpp>
#include "impl/dispatch/system_memory_manager.hpp"

namespace tt::tt_metal {

// PinnedMemoryImpl implementation
PinnedMemoryImpl::PinnedMemoryImpl(
    const std::vector<IDevice*>& devices, void* host_buffer, size_t buffer_size, bool map_to_noc) :
    buffer_size_(buffer_size), map_to_noc_(map_to_noc) {
    initialize_from_devices(devices, host_buffer, buffer_size, map_to_noc);
}

PinnedMemoryImpl::~PinnedMemoryImpl() { device_buffers_.clear(); }

PinnedMemoryImpl::PinnedMemoryImpl(PinnedMemoryImpl&& other) noexcept :
    buffer_size_(other.buffer_size_),
    map_to_noc_(other.map_to_noc_),
    host_offset_(other.host_offset_),
    device_buffers_(std::move(other.device_buffers_)),
    device_to_mmio_map_(std::move(other.device_to_mmio_map_)) {
    other.buffer_size_ = 0;
    other.map_to_noc_ = false;
    other.host_offset_ = 0;
}

PinnedMemoryImpl& PinnedMemoryImpl::operator=(PinnedMemoryImpl&& other) noexcept {
    if (this != &other) {
        device_buffers_.clear();

        buffer_size_ = other.buffer_size_;
        map_to_noc_ = other.map_to_noc_;
        host_offset_ = other.host_offset_;
        device_buffers_ = std::move(other.device_buffers_);
        device_to_mmio_map_ = std::move(other.device_to_mmio_map_);

        other.buffer_size_ = 0;
        other.map_to_noc_ = false;
        other.host_offset_ = 0;
    }
    return *this;
}

void PinnedMemoryImpl::initialize_from_devices(
    const std::vector<IDevice*>& devices, void* host_buffer, size_t buffer_size, bool map_to_noc) {
    if (devices.empty()) {
        throw std::invalid_argument("Cannot create PinnedMemory with empty device list");
    }

    if (buffer_size == 0) {
        throw std::invalid_argument("Buffer size must be greater than 0");
    }

    if (!host_buffer) {
        throw std::invalid_argument(
            "PinnedMemory only supports mapping existing host memory. Use constructor with host_buffer parameter.");
    }

    auto& cluster = MetalContext::instance().get_cluster();

    // Collect all devices and their associated MMIO devices, deduplicating MMIO devices
    std::unordered_map<chip_id_t, chip_id_t> device_to_mmio_map;
    std::unordered_set<chip_id_t> unique_mmio_devices;

    for (IDevice* device : devices) {
        chip_id_t device_id = device->id();
        chip_id_t mmio_device_id = cluster.get_associated_mmio_device(device_id);
        device_to_mmio_map[device_id] = mmio_device_id;
        unique_mmio_devices.insert(mmio_device_id);
    }

    // Determine system page size and align the host buffer down to page boundary
    size_t page_size = 4096;
    long sys_page = sysconf(_SC_PAGESIZE);
    if (sys_page > 0) {
        page_size = static_cast<size_t>(sys_page);
    }

    uintptr_t host_addr = reinterpret_cast<uintptr_t>(host_buffer);
    uintptr_t aligned_base_addr = host_addr & ~(static_cast<uintptr_t>(page_size) - 1);
    host_offset_ = static_cast<size_t>(host_addr - aligned_base_addr);
    size_t mapped_size = buffer_size + host_offset_;
    void* aligned_host_ptr = reinterpret_cast<void*>(aligned_base_addr);

    // Create one buffer per unique MMIO device, all mapping the same aligned host memory
    std::unordered_map<chip_id_t, std::unique_ptr<tt::umd::SysmemBuffer>> mmio_buffers;

    for (chip_id_t mmio_device_id : unique_mmio_devices) {
        auto buffer = cluster.map_sysmem_buffer(mmio_device_id, aligned_host_ptr, mapped_size, map_to_noc);

        if (!buffer) {
            throw std::runtime_error("Failed to create SysmemBuffer for MMIO device " + std::to_string(mmio_device_id));
        }

        mmio_buffers[mmio_device_id] = std::move(buffer);
    }

    device_buffers_ = std::move(mmio_buffers);
    device_to_mmio_map_ = std::move(device_to_mmio_map);
}

tt::umd::SysmemBuffer& PinnedMemoryImpl::get_buffer(chip_id_t device_id) {
    auto mmio_it = device_to_mmio_map_.find(device_id);
    if (mmio_it == device_to_mmio_map_.end()) {
        throw std::invalid_argument("Device " + std::to_string(device_id) + " not found in PinnedMemory");
    }

    chip_id_t mmio_device_id = mmio_it->second;
    auto buffer_it = device_buffers_.find(mmio_device_id);
    if (buffer_it == device_buffers_.end()) {
        throw std::invalid_argument(
            "MMIO device " + std::to_string(mmio_device_id) + " buffer not found in PinnedMemory");
    }
    return *buffer_it->second;
}

const tt::umd::SysmemBuffer& PinnedMemoryImpl::get_buffer(chip_id_t device_id) const {
    auto mmio_it = device_to_mmio_map_.find(device_id);
    if (mmio_it == device_to_mmio_map_.end()) {
        throw std::invalid_argument("Device " + std::to_string(device_id) + " not found in PinnedMemory");
    }

    chip_id_t mmio_device_id = mmio_it->second;
    auto buffer_it = device_buffers_.find(mmio_device_id);
    if (buffer_it == device_buffers_.end()) {
        throw std::invalid_argument(
            "MMIO device " + std::to_string(mmio_device_id) + " buffer not found in PinnedMemory");
    }
    return *buffer_it->second;
}

void* PinnedMemoryImpl::get_host_ptr() {
    if (device_buffers_.empty()) {
        throw std::runtime_error("No buffers available in PinnedMemory");
    }
    // Return the original (unaligned) host pointer by adjusting from aligned base
    auto* base = static_cast<std::uint8_t*>(device_buffers_.begin()->second->get_buffer_va());
    return static_cast<void*>(base + host_offset_);
}

const void* PinnedMemoryImpl::get_host_ptr() const {
    if (device_buffers_.empty()) {
        throw std::runtime_error("No buffers available in PinnedMemory");
    }
    // Return the original (unaligned) host pointer by adjusting from aligned base
    auto* base = static_cast<const std::uint8_t*>(device_buffers_.begin()->second->get_buffer_va());
    return static_cast<const void*>(base + host_offset_);
}

uint64_t PinnedMemoryImpl::get_device_addr(chip_id_t device_id) const {
    return get_buffer(device_id).get_device_io_addr() + static_cast<uint64_t>(host_offset_);
}

std::optional<PinnedMemory::NocAddr> PinnedMemoryImpl::get_noc_addr(chip_id_t device_id) const {
    auto mmio_it = device_to_mmio_map_.find(device_id);
    if (mmio_it == device_to_mmio_map_.end()) {
        return std::nullopt;
    }

    chip_id_t mmio_device_id = mmio_it->second;
    auto buffer_it = device_buffers_.find(mmio_device_id);
    if (buffer_it == device_buffers_.end()) {
        return std::nullopt;
    }

    auto noc_addr_opt = buffer_it->second->get_noc_addr();
    if (!noc_addr_opt.has_value()) {
        return std::nullopt;
    }
    // fmt::println(stderr, "noc_addr_opt: {}", noc_addr_opt.value());

    // Return NOC address and the MMIO device ID where it's usable from
    return PinnedMemory::NocAddr{noc_addr_opt.value() + static_cast<uint64_t>(host_offset_), mmio_device_id};
}

std::vector<chip_id_t> PinnedMemoryImpl::get_device_ids() const {
    std::vector<chip_id_t> device_ids;
    device_ids.reserve(device_to_mmio_map_.size());

    for (const auto& pair : device_to_mmio_map_) {
        device_ids.push_back(pair.first);
    }

    std::sort(device_ids.begin(), device_ids.end());
    return device_ids;
}

bool PinnedMemoryImpl::has_device(chip_id_t device_id) const {
    return device_to_mmio_map_.find(device_id) != device_to_mmio_map_.end();
}

bool PinnedMemoryImpl::usable_from_noc(chip_id_t device_id) const {
    // Check if mapped to NOC and device is its own MMIO device (i.e., MMIO-capable)
    auto mmio_it = device_to_mmio_map_.find(device_id);
    if (mmio_it == device_to_mmio_map_.end()) {
        return false;
    }

    return map_to_noc_ && (mmio_it->second == device_id);
}

void PinnedMemoryImpl::add_barrier_event(const distributed::MeshEvent& event) {
    barrier_events_.push_back(event);

    while (!barrier_events_.empty()) {
        auto& event = barrier_events_.front();
        if (!tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch()) {
            barrier_events_.pop_front();
            continue;
        }
        bool all_devices_completed = true;
        for (const auto& coord : event.device_range()) {
            auto physical_device = event.device()->get_device(coord);
            if (physical_device->sysmem_manager().get_last_completed_event(event.mesh_cq_id()) < event.id()) {
                all_devices_completed = false;
                break;
            }
        }
        if (all_devices_completed) {
            barrier_events_.pop_front();
        } else {
            break;
        }
    }
}

void* PinnedMemoryImpl::lock() {
    while (!barrier_events_.empty()) {
        auto& event = barrier_events_.front();
        distributed::EventSynchronize(event);
        barrier_events_.pop_front();
    }
    return get_host_ptr();
}

void PinnedMemoryImpl::unlock() {}

// PinnedMemory pimpl wrapper implementation
PinnedMemory::PinnedMemory(
    const std::vector<IDevice*>& devices, void* host_buffer, size_t buffer_size, bool map_to_noc) :
    pImpl(std::make_unique<PinnedMemoryImpl>(devices, host_buffer, buffer_size, map_to_noc)) {}

PinnedMemory::~PinnedMemory() = default;

PinnedMemory::PinnedMemory(PinnedMemory&& other) noexcept = default;
PinnedMemory& PinnedMemory::operator=(PinnedMemory&& other) noexcept = default;

tt::umd::SysmemBuffer& PinnedMemory::get_buffer(chip_id_t device_id) { return pImpl->get_buffer(device_id); }

const tt::umd::SysmemBuffer& PinnedMemory::get_buffer(chip_id_t device_id) const {
    return pImpl->get_buffer(device_id);
}

void* PinnedMemory::get_host_ptr() { return pImpl->get_host_ptr(); }

const void* PinnedMemory::get_host_ptr() const { return pImpl->get_host_ptr(); }

uint64_t PinnedMemory::get_device_addr(chip_id_t device_id) const { return pImpl->get_device_addr(device_id); }

std::optional<PinnedMemory::NocAddr> PinnedMemory::get_noc_addr(chip_id_t device_id) const {
    return pImpl->get_noc_addr(device_id);
}

size_t PinnedMemory::get_buffer_size() const { return pImpl->get_buffer_size(); }

std::vector<chip_id_t> PinnedMemory::get_device_ids() const { return pImpl->get_device_ids(); }

bool PinnedMemory::has_device(chip_id_t device_id) const { return pImpl->has_device(device_id); }

bool PinnedMemory::usable_from_noc(chip_id_t device_id) const { return pImpl->usable_from_noc(device_id); }

void PinnedMemory::add_barrier_event(const distributed::MeshEvent& event) { pImpl->add_barrier_event(event); }

void* PinnedMemory::lock() { return pImpl->lock(); }

void PinnedMemory::unlock() { pImpl->unlock(); }

}  // namespace tt::tt_metal
