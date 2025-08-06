// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/pinned_memory.hpp>

#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <memory>

#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <context/metal_context.hpp>
#include <umd/device/chip_helpers/sysmem_manager.h>
#include <umd/device/chip_helpers/sysmem_buffer.h>

namespace tt::tt_metal {

PinnedMemory::PinnedMemory(
    const std::vector<IDevice*>& devices,
    size_t buffer_size,
    bool map_to_noc)
    : buffer_size_(buffer_size), map_to_noc_(map_to_noc), owns_host_memory_(true), host_memory_base_(nullptr) {
    
    initialize_from_devices(devices, nullptr, buffer_size, map_to_noc);
}

PinnedMemory::PinnedMemory(
    const std::vector<IDevice*>& devices,
    void* host_buffer,
    size_t buffer_size,
    bool map_to_noc)
    : buffer_size_(buffer_size), map_to_noc_(map_to_noc), owns_host_memory_(false), host_memory_base_(host_buffer) {
    
    initialize_from_devices(devices, host_buffer, buffer_size, map_to_noc);
}

PinnedMemory::~PinnedMemory() {
    // SysmemBuffers will be automatically cleaned up by their destructors
    device_buffers_.clear();
}

PinnedMemory::PinnedMemory(PinnedMemory&& other) noexcept
    : buffer_size_(other.buffer_size_)
    , map_to_noc_(other.map_to_noc_)
    , owns_host_memory_(other.owns_host_memory_)
    , host_memory_base_(other.host_memory_base_)
    , device_buffers_(std::move(other.device_buffers_)) {
    
    // Reset the other object
    other.buffer_size_ = 0;
    other.map_to_noc_ = false;
    other.owns_host_memory_ = false;
    other.host_memory_base_ = nullptr;
}

PinnedMemory& PinnedMemory::operator=(PinnedMemory&& other) noexcept {
    if (this != &other) {
        // Clean up current resources
        device_buffers_.clear();
        
        // Move from other
        buffer_size_ = other.buffer_size_;
        map_to_noc_ = other.map_to_noc_;
        owns_host_memory_ = other.owns_host_memory_;
        host_memory_base_ = other.host_memory_base_;
        device_buffers_ = std::move(other.device_buffers_);
        
        // Reset the other object
        other.buffer_size_ = 0;
        other.map_to_noc_ = false;
        other.owns_host_memory_ = false;
        other.host_memory_base_ = nullptr;
    }
    return *this;
}

void PinnedMemory::initialize_from_devices(
    const std::vector<IDevice*>& devices,
    void* host_buffer,
    size_t buffer_size,
    bool map_to_noc) {
    
    if (devices.empty()) {
        throw std::invalid_argument("Cannot create PinnedMemory with empty device list");
    }
    
    if (buffer_size == 0) {
        throw std::invalid_argument("Buffer size must be greater than 0");
    }
    
    // Get the cluster to access SysmemManagers
    auto& cluster = MetalContext::instance().get_cluster();
    
    size_t device_offset = 0;
    for (IDevice* device : devices) {
        chip_id_t device_id = device->id();
        
        // Use the new cluster API methods to access SysmemManager functionality
        std::unique_ptr<tt::umd::SysmemBuffer> buffer;
        
        if (host_buffer) {
            // Map existing host memory
            void* device_buffer_ptr = static_cast<char*>(host_buffer) + device_offset;
            buffer = cluster.map_sysmem_buffer(device_id, device_buffer_ptr, buffer_size, map_to_noc);
        } else {
            // Allocate new memory
            buffer = cluster.allocate_sysmem_buffer(device_id, buffer_size, map_to_noc);
        }
        
        if (!buffer) {
            throw std::runtime_error("Failed to create SysmemBuffer for device " + std::to_string(device_id));
        }
        
        device_buffers_[device_id] = std::move(buffer);
        device_offset += buffer_size;
    }
    
    // If we allocated our own memory, store the base pointer from the first buffer
    if (!host_buffer && !device_buffers_.empty()) {
        host_memory_base_ = device_buffers_.begin()->second->get_buffer_va();
    }
}

tt::umd::SysmemBuffer& PinnedMemory::get_buffer(chip_id_t device_id) {
    auto it = device_buffers_.find(device_id);
    if (it == device_buffers_.end()) {
        throw std::invalid_argument("Device " + std::to_string(device_id) + " not found in PinnedMemory");
    }
    return *it->second;
}

const tt::umd::SysmemBuffer& PinnedMemory::get_buffer(chip_id_t device_id) const {
    auto it = device_buffers_.find(device_id);
    if (it == device_buffers_.end()) {
        throw std::invalid_argument("Device " + std::to_string(device_id) + " not found in PinnedMemory");
    }
    return *it->second;
}

void* PinnedMemory::get_host_ptr(chip_id_t device_id) {
    return get_buffer(device_id).get_buffer_va();
}

const void* PinnedMemory::get_host_ptr(chip_id_t device_id) const {
    return get_buffer(device_id).get_buffer_va();
}

uint64_t PinnedMemory::get_device_addr(chip_id_t device_id) const {
    return get_buffer(device_id).get_device_io_addr();
}

std::vector<chip_id_t> PinnedMemory::get_device_ids() const {
    std::vector<chip_id_t> device_ids;
    device_ids.reserve(device_buffers_.size());
    
    for (const auto& pair : device_buffers_) {
        device_ids.push_back(pair.first);
    }
    
    std::sort(device_ids.begin(), device_ids.end());
    return device_ids;
}

bool PinnedMemory::has_device(chip_id_t device_id) const {
    return device_buffers_.find(device_id) != device_buffers_.end();
}

void PinnedMemory::write_to_device(chip_id_t device_id, const void* src, size_t size, size_t offset) {
    if (offset + size > buffer_size_) {
        throw std::invalid_argument("Write operation exceeds buffer size");
    }
    
    void* dest_ptr = static_cast<char*>(get_host_ptr(device_id)) + offset;
    std::memcpy(dest_ptr, src, size);
}

void PinnedMemory::read_from_device(chip_id_t device_id, void* dest, size_t size, size_t offset) {
    if (offset + size > buffer_size_) {
        throw std::invalid_argument("Read operation exceeds buffer size");
    }
    
    const void* src_ptr = static_cast<const char*>(get_host_ptr(device_id)) + offset;
    std::memcpy(dest, src_ptr, size);
}

}  // namespace tt::tt_metal 