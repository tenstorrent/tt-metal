// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/dmabuf_export.hpp>

#include <unistd.h>

#include <algorithm>
#include <utility>

#include <tt_stl/assert.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_align.hpp>
#include <umd/device/arch/architecture_tlbs.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/pcie/pci_device.hpp>

#include "tt_metal/impl/context/context_types.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal::experimental::dmabuf_export {

namespace {

struct ResolvedBank {
    const tt::Cluster* cluster = nullptr;
    ChipId chip_id = 0;
    tt::ARCH arch = tt::ARCH::Invalid;
    tt::umd::CoreCoord core;
    // NOC address, within `core`, of the extent's first byte.
    uint64_t address = 0;
};

uint64_t host_page_size() { return static_cast<uint64_t>(getpagesize()); }

ResolvedBank resolve_bank(const distributed::MeshBuffer& mesh_buffer, const DramBankExtent& extent) {
    TT_FATAL(mesh_buffer.is_allocated(), "MeshBuffer must be allocated to resolve its DRAM banks");
    IDevice* device = mesh_buffer.get_device_buffer(extent.device_coord)->device();
    const tt::Cluster& cluster = MetalContext::instance(extract_context_id(device)).get_cluster();
    const metal_SocDescriptor& soc_desc = cluster.get_soc_desc(device->id());

    const int dram_view = static_cast<int>(extent.bank_id);
    TT_FATAL(
        dram_view < static_cast<int>(soc_desc.get_num_dram_views()),
        "DRAM bank {} has no matching DRAM view on device {} ({} views)",
        extent.bank_id,
        device->id(),
        soc_desc.get_num_dram_views());
    const CoreCoord worker_core = soc_desc.get_preferred_worker_core_for_dram_view(dram_view, NOC::NOC_0);

    return ResolvedBank{
        .cluster = &cluster,
        .chip_id = device->id(),
        .arch = device->arch(),
        .core = soc_desc.get_coord_at(worker_core, CoordSystem::TRANSLATED),
        .address = extent.address + soc_desc.get_address_offset(dram_view),
    };
}

DeviceAddr max_export_size_at(const ResolvedBank& bank, const DramBankExtent& extent, DeviceAddr offset) {
    const uint64_t start = bank.address + offset;
    const uint64_t covering_start = start - (start % host_page_size());
    const uint64_t window_size = tt::umd::get_architecture_tlbs(bank.arch).size_classes.back().size;
    const uint64_t window_end = covering_start - (covering_start % window_size) + window_size;
    return std::min<DeviceAddr>(window_end - start, extent.size() - offset);
}

}  // namespace

bool is_supported() { return tt::umd::PCIDevice::is_tlb_dmabuf_export_supported(); }

std::vector<DramBankExtent> get_dram_bank_extents(const distributed::MeshBuffer& mesh_buffer) {
    TT_FATAL(mesh_buffer.is_allocated(), "MeshBuffer must be allocated to compute its DRAM bank extents");
    const Buffer& reference = *mesh_buffer.get_reference_buffer();
    TT_FATAL(
        reference.buffer_type() == BufferType::DRAM,
        "dma-buf export supports DRAM buffers only, got {}",
        reference.buffer_type());
    TT_FATAL(
        reference.buffer_layout() == TensorMemoryLayout::INTERLEAVED,
        "dma-buf export supports interleaved buffers only, got {}",
        reference.buffer_layout());

    const uint32_t num_banks = reference.allocator()->get_num_banks(BufferType::DRAM);
    const uint32_t num_pages = reference.num_pages();
    const uint32_t num_used_banks = std::min(num_banks, num_pages);

    distributed::MeshDevice* mesh_device = mesh_buffer.device();
    std::vector<DramBankExtent> extents;
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        if (!mesh_device->is_local(coord)) {
            continue;
        }
        for (uint32_t bank_id = 0; bank_id < num_used_banks; ++bank_id) {
            extents.push_back(DramBankExtent{
                .device_coord = coord,
                .bank_id = bank_id,
                .first_page = bank_id,
                .page_stride = num_banks,
                .num_pages = (num_pages - bank_id + num_banks - 1) / num_banks,
                .page_size = reference.page_size(),
                .aligned_page_size = reference.aligned_page_size(),
                .address = reference.address(),
            });
        }
    }
    return extents;
}

DeviceAddr max_export_size(const distributed::MeshBuffer& mesh_buffer, const DramBankExtent& extent, DeviceAddr offset) {
    TT_FATAL(offset < extent.size(), "Offset {} is outside the {} byte DRAM bank extent", offset, extent.size());
    return max_export_size_at(resolve_bank(mesh_buffer, extent), extent, offset);
}

DmabufExport export_dram_bank_range(
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    const DramBankExtent& extent,
    DeviceAddr offset,
    DeviceAddr size,
    ExportBounds bounds) {
    TT_FATAL(mesh_buffer != nullptr, "Cannot export a null MeshBuffer");
    TT_FATAL(size > 0, "Cannot export an empty range");
    TT_FATAL(
        offset + size <= extent.size(),
        "Range [{}, {}) is outside the {} byte DRAM bank extent",
        offset,
        offset + size,
        extent.size());

    const ResolvedBank bank = resolve_bank(*mesh_buffer, extent);
    TT_FATAL(
        bank.cluster->get_associated_mmio_device(bank.chip_id) == bank.chip_id,
        "Device {} at {} has no PCIe link of its own, so its memory cannot be exported as a dma-buf",
        bank.chip_id,
        extent.device_coord);
    const DeviceAddr max_size = max_export_size_at(bank, extent, offset);
    TT_FATAL(
        size <= max_size,
        "{} bytes at offset {} do not fit in one device window; at most {} bytes fit, see max_export_size",
        size,
        offset,
        max_size);

    const uint64_t page = host_page_size();
    const uint64_t start = bank.address + offset;
    const uint64_t end = start + size;
    uint64_t export_start = start;
    uint64_t export_end = end;
    if (bounds == ExportBounds::PageCovering) {
        export_start = start - (start % page);
        export_end = tt::align(end, page);
    } else {
        TT_FATAL(
            start % page == 0 && end % page == 0,
            "Range [{:#x}, {:#x}) on device {} is not aligned to the {} byte host page; use ExportBounds::PageCovering",
            start,
            end,
            bank.chip_id,
            page);
    }

    const int fd =
        bank.cluster->get_driver()->export_dmabuf(bank.chip_id, bank.core, export_start, export_end - export_start);
    return DmabufExport(fd, export_end - export_start, start - export_start, size, mesh_buffer);
}

void wait_for_device_idle(const distributed::MeshBuffer& mesh_buffer) {
    distributed::MeshDevice* mesh_device = mesh_buffer.device();
    for (uint8_t cq_id = 0; cq_id < mesh_device->num_hw_cqs(); ++cq_id) {
        distributed::Finish(mesh_device->mesh_command_queue(cq_id));
    }
}

void flush_external_writes(const distributed::MeshBuffer& mesh_buffer) {
    distributed::MeshDevice* mesh_device = mesh_buffer.device();
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        if (!mesh_device->is_local(coord)) {
            continue;
        }
        IDevice* device = mesh_buffer.get_device_buffer(coord)->device();
        MetalContext::instance(extract_context_id(device)).get_cluster().dram_barrier(device->id());
    }
}

DmabufExport::DmabufExport(
    int fd, size_t size, size_t data_offset, size_t data_size, std::shared_ptr<distributed::MeshBuffer> mesh_buffer) :
    fd_(fd), size_(size), data_offset_(data_offset), data_size_(data_size), mesh_buffer_(std::move(mesh_buffer)) {}

DmabufExport::DmabufExport(DmabufExport&& other) noexcept :
    fd_(std::exchange(other.fd_, -1)),
    size_(std::exchange(other.size_, 0)),
    data_offset_(std::exchange(other.data_offset_, 0)),
    data_size_(std::exchange(other.data_size_, 0)),
    mesh_buffer_(std::move(other.mesh_buffer_)) {}

DmabufExport& DmabufExport::operator=(DmabufExport&& other) noexcept {
    if (this != &other) {
        reset();
        fd_ = std::exchange(other.fd_, -1);
        size_ = std::exchange(other.size_, 0);
        data_offset_ = std::exchange(other.data_offset_, 0);
        data_size_ = std::exchange(other.data_size_, 0);
        mesh_buffer_ = std::move(other.mesh_buffer_);
    }
    return *this;
}

DmabufExport::~DmabufExport() { reset(); }

void DmabufExport::reset() noexcept {
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
    mesh_buffer_.reset();
}

}  // namespace tt::tt_metal::experimental::dmabuf_export
