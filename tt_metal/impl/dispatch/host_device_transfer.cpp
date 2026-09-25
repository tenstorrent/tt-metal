// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/dispatch/host_device_transfer.hpp"

#include <cstring>

#include <tt_stl/assert.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>

#include "impl/allocator/allocator.hpp"
#include "impl/context/context_types.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/emulation/host_sanitizers.hpp"
#include "llrt/tt_cluster.hpp"
#include "tracy/Tracy.hpp"
#include "tt_align.hpp"

namespace tt::tt_metal::slow_dispatch {

bool WriteToDeviceDRAMChannel(
    IDevice& device, int dram_channel, uint32_t address, std::span<const uint8_t> host_buffer) {
    if constexpr (emule::kEmuleAsanBuild) {
        emule::check_host_dram_alignment(
            &device, address, static_cast<uint32_t>(host_buffer.size()), "WriteToDeviceDRAMChannel");
    }
    TT_FATAL(
        address >= device.allocator()->get_base_allocator_addr(HalMemType::DRAM),
        "Cannot write to reserved DRAM region, addresses [0, {}) are reserved!",
        device.allocator()->get_base_allocator_addr(HalMemType::DRAM));
    const MetalContext& metal_ctx = MetalContext::instance(extract_context_id(&device));
    metal_ctx.get_cluster().write_dram_vec(
        host_buffer.data(), host_buffer.size(), device.id(), dram_channel, address, tt::umd::IoOrdering::Relaxed);
    return true;
}

bool WriteToDeviceDRAMChannel(IDevice& device, int dram_channel, uint32_t address, std::vector<uint32_t>& host_buffer) {
    return WriteToDeviceDRAMChannel(
        device,
        dram_channel,
        address,
        std::span(reinterpret_cast<const uint8_t*>(host_buffer.data()), host_buffer.size() * sizeof(uint32_t)));
}

bool ReadFromDeviceDRAMChannel(IDevice& device, int dram_channel, uint32_t address, std::span<uint8_t> host_buffer) {
    if constexpr (emule::kEmuleAsanBuild) {
        emule::check_host_dram_alignment(
            &device, address, static_cast<uint32_t>(host_buffer.size()), "ReadFromDeviceDRAMChannel");
    }
    const MetalContext& metal_ctx = MetalContext::instance(extract_context_id(&device));
    metal_ctx.get_cluster().dram_barrier(device.id());
    metal_ctx.get_cluster().read_dram_vec(host_buffer.data(), host_buffer.size(), device.id(), dram_channel, address);
    return true;
}

bool ReadFromDeviceDRAMChannel(
    IDevice& device, int dram_channel, uint32_t address, uint32_t size, std::vector<uint32_t>& host_buffer) {
    host_buffer.resize((size + sizeof(uint32_t) - 1) / sizeof(uint32_t));
    return ReadFromDeviceDRAMChannel(
        device, dram_channel, address, std::span(reinterpret_cast<uint8_t*>(host_buffer.data()), size));
}

namespace {

using experimental::per_core_allocation::get_shard_base_address;

void WriteToDeviceSharded(
    Buffer& buffer, ttsl::Span<const uint8_t> host_buffer, const CoreRangeSet* logical_core_filter) {
    TT_FATAL(
        host_buffer.size() <= buffer.size(),
        "Bounds-Error -- Attempting to write {} bytes to a {} byte buffer",
        host_buffer.size(),
        buffer.size());

    uint32_t page_size = buffer.page_size();
    TT_ASSERT(page_size == 0 ? buffer.size() == 0 : buffer.size() % page_size == 0);

    auto* device = buffer.device();
    const auto& allocator = device->allocator();

    const MetalContext& metal_ctx = MetalContext::instance(extract_context_id(device));
    const auto& cluster = metal_ctx.get_cluster();
    const size_t alignment_req = cluster.get_alignment_requirements(device->id(), page_size);
    const size_t aligned_bytes = alignment_req ? (page_size / alignment_req) * alignment_req : page_size;
    const size_t remainder_bytes = page_size - aligned_bytes;
    TT_ASSERT(buffer.aligned_page_size() >= page_size);  // Check that we don't write to the end of the buffer
    const auto& buffer_page_mapping = *buffer.get_buffer_page_mapping();
    const bool can_write_page_ranges = buffer.aligned_page_size() == page_size;

    auto write_pages = [&](uint32_t core_id, uint32_t device_page, uint32_t host_page, uint32_t num_pages) {
        if (num_pages == 0) {
            return;
        }
        auto core = buffer_page_mapping.all_cores[core_id];
        if (logical_core_filter != nullptr && !logical_core_filter->contains(core)) {
            return;
        }
        auto bank_id = allocator->get_bank_ids_from_logical_core(buffer.buffer_type(), core)[0];
        auto bank_offset = allocator->get_bank_offset(buffer.buffer_type(), bank_id);
        size_t data_index = static_cast<size_t>(host_page) * page_size;
        auto write_chunk = [&](uint32_t write_device_page, size_t offset, size_t size_in_bytes) {
            if (size_in_bytes == 0) {
                return;
            }
            std::span<const std::uint8_t> page(host_buffer.data() + data_index + offset, size_in_bytes);
            if (buffer.is_l1()) {
                auto absolute_address = get_shard_base_address(buffer, core) + bank_offset +
                                        (write_device_page * buffer.aligned_page_size()) + offset;
                auto core_coordinates =
                    device->worker_core_from_logical_core(buffer.allocator()->get_logical_core_from_bank_id(bank_id));
                cluster.write_core(device->id(), core_coordinates, page, absolute_address);
            } else {
                auto bank_local_address = buffer.address() + (write_device_page * buffer.aligned_page_size()) + offset;
                WriteToDeviceDRAMChannel(*device, bank_id, bank_local_address, page);
            }
        };

        if (can_write_page_ranges) {
            write_chunk(device_page, 0, static_cast<size_t>(num_pages) * page_size);
            return;
        }

        for (uint32_t page = 0; page < num_pages; page++) {
            data_index = static_cast<size_t>(host_page + page) * page_size;
            write_chunk(device_page + page, 0, aligned_bytes);
            write_chunk(device_page + page, aligned_bytes, remainder_bytes);
        }
    };

    for (uint32_t core_id = 0; core_id < buffer_page_mapping.all_cores.size(); core_id++) {
        for (const auto& core_page_mapping : buffer_page_mapping.core_page_mappings[core_id]) {
            for (const auto& host_range : core_page_mapping.host_ranges) {
                write_pages(
                    core_id,
                    core_page_mapping.device_start_page + host_range.device_page_offset,
                    host_range.host_page_start,
                    host_range.num_pages);
            }
        }
    }
}

DeviceAddr CalculateAddressDeviceInterleavedContiguous(const Buffer& buffer, uint64_t bank_index, uint64_t page_index) {
    DeviceAddr addr = 0;
    if (buffer.is_dram()) {
        uint32_t num_banks = buffer.allocator()->get_num_banks(buffer.buffer_type());
        uint32_t pages_offset_within_bank = page_index / num_banks;
        addr = buffer.address() + pages_offset_within_bank * buffer.aligned_page_size();
    } else {
        TT_ASSERT(buffer.is_l1());
        addr = buffer.page_address(bank_index, page_index);
    }

    return addr;
}

void WriteToDeviceInterleavedContiguous(const Buffer& buffer, ttsl::Span<const uint8_t> host_buffer) {
    if (GraphTracker::instance().hook_write_to_device(&buffer)) {
        return;
    }

    size_t host_buffer_size_bytes = host_buffer.size();
    TT_FATAL(
        host_buffer_size_bytes <= buffer.size(),
        "Bounds-Error -- Attempting to write {} bytes to a {} byte buffer",
        host_buffer_size_bytes,
        buffer.size());

    size_t page_size = buffer.page_size();
    size_t num_pages = buffer.num_pages();

    auto* device = buffer.device();
    size_t num_banks = device->allocator()->get_num_banks(buffer.buffer_type());
    size_t bank_index = 0;
    size_t data_index = 0;

    const MetalContext& metal_ctx = MetalContext::instance(extract_context_id(device));
    const auto& cluster = metal_ctx.get_cluster();
    const size_t alignment_req = cluster.get_alignment_requirements(device->id(), page_size);
    const size_t aligned_bytes = alignment_req ? (page_size / alignment_req) * alignment_req : page_size;
    const size_t remainder_bytes = page_size - aligned_bytes;
    TT_ASSERT(buffer.aligned_page_size() >= page_size);  // Check that we don't write to the end of the buffer
    for (size_t page_index = 0; page_index < num_pages; page_index++) {
        const DeviceAddr address = CalculateAddressDeviceInterleavedContiguous(buffer, bank_index, page_index);
        auto write_chunk = [&](size_t offset, size_t size_in_bytes) {
            if (size_in_bytes == 0) {
                return;
            }
            std::span<const std::uint8_t> page(host_buffer.data() + data_index + offset, size_in_bytes);
            switch (buffer.buffer_type()) {
                case BufferType::DRAM: WriteToDeviceDRAMChannel(*device, bank_index, address + offset, page); break;
                case BufferType::L1:
                case BufferType::L1_SMALL: {
                    CoreCoord logical_core = buffer.allocator()->get_logical_core_from_bank_id(bank_index);
                    detail::WriteToDeviceL1(device, logical_core, address + offset, page, CoreType::WORKER);
                } break;
                default: TT_THROW("Unsupported buffer type to write to device!");
            }
        };

        write_chunk(0, aligned_bytes);
        write_chunk(aligned_bytes, remainder_bytes);

        bank_index = (bank_index + 1) % num_banks;
        data_index += page_size;
    }
}

void WriteToDevice(Buffer& buffer, ttsl::Span<const uint8_t> host_buffer, const CoreRangeSet* logical_core_filter) {
    ZoneScoped;
    if (buffer.buffer_layout() == TensorMemoryLayout::INTERLEAVED) {
        if (logical_core_filter != nullptr) {
            TT_FATAL(
                logical_core_filter->empty(),
                "logical_core_filter is only supported for sharded buffer layouts (interleaved layout does not support "
                "per-core filtering)");
            return;
        }
        WriteToDeviceInterleavedContiguous(buffer, host_buffer);
    } else if (is_sharded(buffer.buffer_layout())) {
        WriteToDeviceSharded(buffer, host_buffer, logical_core_filter);
    } else {
        TT_ASSERT(false && "Unsupported buffer layout");
    }
}

void ReadFromDeviceInterleavedContiguous(const Buffer& buffer, uint8_t* host_buffer) {
    size_t page_size = buffer.page_size();
    size_t num_pages = buffer.num_pages();

    auto* device = buffer.device();
    size_t num_banks = device->allocator()->get_num_banks(buffer.buffer_type());

    size_t host_idx = 0;
    size_t bank_index = 0;

    const MetalContext& metal_ctx = MetalContext::instance(extract_context_id(device));
    const auto& cluster = metal_ctx.get_cluster();
    size_t aligned_page_size = tt::align(page_size, cluster.get_alignment_requirements(device->id(), page_size));

    std::vector<uint8_t> page(aligned_page_size);
    for (size_t page_index = 0; page_index < num_pages; page_index++) {
        const DeviceAddr address = CalculateAddressDeviceInterleavedContiguous(buffer, bank_index, page_index);
        switch (buffer.buffer_type()) {
            case BufferType::DRAM:
            case BufferType::TRACE: {
                ReadFromDeviceDRAMChannel(*device, bank_index, address, std::span<uint8_t>(page));
            } break;
            case BufferType::L1:
            case BufferType::L1_SMALL: {
                auto core_coordinates = device->worker_core_from_logical_core(
                    buffer.allocator()->get_logical_core_from_bank_id(bank_index));
                cluster.read_core(page.data(), aligned_page_size, tt_cxy_pair(device->id(), core_coordinates), address);
            } break;
            default: TT_THROW("Unsupported buffer type to read from device!");
        }

        // Copy page into host buffer
        std::memcpy(host_buffer + host_idx, page.data(), page_size);
        host_idx += page_size;

        bank_index = (bank_index + 1) % num_banks;
    }
}

void read_pages_to_host_helper(
    IDevice* device,
    Buffer& dev_buffer,
    uint8_t* host_buffer,
    const uint32_t& page_size,
    const uint32_t& host_page_id,
    const uint32_t& core_page_id,
    const uint32_t& bank_id) {
    uint64_t host_buffer_start = uint64_t(host_page_id) * page_size;
    const MetalContext& metal_ctx = MetalContext::instance(extract_context_id(device));
    const auto& cluster = metal_ctx.get_cluster();
    size_t aligned_page_size = tt::align(page_size, cluster.get_alignment_requirements(device->id(), page_size));

    if (dev_buffer.is_l1()) {
        auto logical_core = dev_buffer.allocator()->get_logical_core_from_bank_id(bank_id);
        auto core_coordinates = device->worker_core_from_logical_core(logical_core);
        auto bank_offset = device->allocator()->get_bank_offset(dev_buffer.buffer_type(), bank_id);
        auto absolute_address = get_shard_base_address(dev_buffer, logical_core) + bank_offset +
                                (core_page_id * dev_buffer.aligned_page_size());
        if (aligned_page_size > page_size) {
            std::vector<uint8_t> page(aligned_page_size);
            cluster.read_core(
                page.data(), aligned_page_size, tt_cxy_pair(device->id(), core_coordinates), absolute_address);
            std::memcpy(host_buffer + host_buffer_start, page.data(), page_size);
        } else {
            cluster.read_core(
                host_buffer + host_buffer_start,
                page_size,
                tt_cxy_pair(device->id(), core_coordinates),
                absolute_address);
        }
    } else {
        std::vector<uint8_t> page(aligned_page_size);
        auto bank_local_address = dev_buffer.address() + (core_page_id * dev_buffer.aligned_page_size());
        ReadFromDeviceDRAMChannel(*device, bank_id, bank_local_address, std::span<uint8_t>(page));
        std::memcpy(host_buffer + host_buffer_start, page.data(), page_size);
    }
}

void ReadFromDeviceSharded(Buffer& buffer, uint8_t* host_buffer) {
    auto* device = buffer.device();

    uint32_t page_size = buffer.page_size();
    const auto& buffer_page_mapping = *buffer.get_buffer_page_mapping();

    for (auto mapped_page : buffer_page_mapping) {
        auto core = buffer_page_mapping.all_cores[mapped_page.core_id];
        auto bank_id = device->allocator()->get_bank_ids_from_logical_core(buffer.buffer_type(), core)[0];
        read_pages_to_host_helper(
            device, buffer, host_buffer, page_size, mapped_page.host_page, mapped_page.device_page, bank_id);
    }
}

void ReadFromDevice(Buffer& buffer, uint8_t* host_buffer) {
    ZoneScoped;
    if (buffer.buffer_layout() == TensorMemoryLayout::INTERLEAVED) {
        ReadFromDeviceInterleavedContiguous(buffer, host_buffer);
    } else if (is_sharded(buffer.buffer_layout())) {
        ReadFromDeviceSharded(buffer, host_buffer);
    } else {
        TT_ASSERT(false && "Unsupported buffer layout");
    }
}

void WriteToBufferImpl(Buffer& buffer, ttsl::Span<const uint8_t> host_buffer, const CoreRangeSet* logical_core_filter) {
    switch (buffer.buffer_type()) {
        case BufferType::DRAM:  // fallthrough
        case BufferType::L1:    // fallthrough
        case BufferType::L1_SMALL: {
            WriteToDevice(buffer, host_buffer, logical_core_filter);
        } break;
        case BufferType::SYSTEM_MEMORY: {
            TT_THROW("Writing to host memory is unsupported!");
        } break;
        default: TT_THROW("Unsupported buffer type!");
    }
}

}  // namespace

void WriteToBuffer(Buffer& buffer, ttsl::Span<const uint8_t> host_buffer) {
    if constexpr (emule::kEmuleAsanBuild) {
        emule::check_buffer_allocated(buffer, "WriteToBuffer");
    }
    WriteToBufferImpl(buffer, host_buffer, /*logical_core_filter=*/nullptr);
}

void WriteToBuffer(Buffer& buffer, ttsl::Span<const uint8_t> host_buffer, const CoreRangeSet& logical_core_filter) {
    if constexpr (emule::kEmuleAsanBuild) {
        emule::check_buffer_allocated(buffer, "WriteToBuffer (core_subset_write)");
    }
    WriteToBufferImpl(buffer, host_buffer, &logical_core_filter);
}

void ReadFromBuffer(Buffer& buffer, uint8_t* host_buffer) {
    if constexpr (emule::kEmuleAsanBuild) {
        emule::check_buffer_allocated(buffer, "ReadFromBuffer");
    }
    IDevice* device = buffer.device();
    switch (buffer.buffer_type()) {
        case BufferType::DRAM:
        case BufferType::TRACE:
        case BufferType::L1:  // fallthrough
        case BufferType::L1_SMALL: {
            const MetalContext& metal_ctx = MetalContext::instance(extract_context_id(device));
            if (buffer.is_dram()) {
                metal_ctx.get_cluster().dram_barrier(device->id());
            } else {
                metal_ctx.get_cluster().l1_barrier(device->id());
            }
            ReadFromDevice(buffer, host_buffer);
        } break;
        case BufferType::SYSTEM_MEMORY: {
            TT_THROW("Reading from host memory is unsupported!");
        } break;
        default: TT_THROW("Unsupported buffer type!");
    }
}

void ReadShard(Buffer& buffer, uint8_t* host_buffer, const uint32_t& core_id) {
    if constexpr (emule::kEmuleAsanBuild) {
        emule::check_buffer_allocated(buffer, "ReadShard");
    }
    IDevice* device = buffer.device();
    TT_ASSERT(is_sharded(buffer.buffer_layout()));

    const auto& buffer_page_mapping = *buffer.get_buffer_page_mapping();
    auto core = buffer_page_mapping.all_cores[core_id];
    auto core_page_mappings = buffer_page_mapping.core_page_mappings[core_id];
    auto bank_id = device->allocator()->get_bank_ids_from_logical_core(buffer.buffer_type(), core)[0];

    if (core_page_mappings.empty()) {
        return;
    }
    size_t shard_offset = core_page_mappings[0].host_ranges[0].host_page_start;

    for (const auto& core_mapping : core_page_mappings) {
        for (auto host_page_it = core_mapping.begin(); host_page_it != core_mapping.end(); host_page_it++) {
            if (!*host_page_it) {
                continue;
            }
            auto host_page_id = **host_page_it - shard_offset;
            auto core_page_id = core_mapping.device_start_page + host_page_it.device_page_offset();
            read_pages_to_host_helper(
                device, buffer, host_buffer, buffer.page_size(), host_page_id, core_page_id, bank_id);
        }
    }
}

}  // namespace tt::tt_metal::slow_dispatch
