// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal::distributed {
class MeshBuffer;
}  // namespace tt::tt_metal::distributed

// EXPERIMENTAL: exports the device DRAM backing a MeshBuffer as Linux dma-buf file descriptors, so a third-party PCIe
// device (an NVMe drive, an RDMA NIC) can DMA into or out of it without staging the data in host memory.
// Only interleaved DRAM buffers are supported.
namespace tt::tt_metal::experimental::dmabuf_export {

// Returns true if the installed KMD and the running kernel can export device memory as a dma-buf.
bool is_supported();

// The part of an interleaved DRAM MeshBuffer that lives in one DRAM bank of one device.
//
// Pages are stored back to back in the bank: the k-th page of the extent is buffer page
// `first_page + k * page_stride`, starting at byte `k * aligned_page_size` of the extent. The bytes between
// `page_size` and `aligned_page_size` of each page are padding.
struct DramBankExtent {
    distributed::MeshCoordinate device_coord;
    uint32_t bank_id = 0;
    uint32_t first_page = 0;
    uint32_t page_stride = 0;
    uint32_t num_pages = 0;
    DeviceAddr page_size = 0;
    DeviceAddr aligned_page_size = 0;
    // Bank-local address of the extent's first byte. Identical for every bank of a MeshBuffer.
    DeviceAddr address = 0;

    DeviceAddr size() const { return static_cast<DeviceAddr>(num_pages) * aligned_page_size; }
};

// Returns the extents of `mesh_buffer` on every device local to this host, ordered by device coordinate, then bank.
// Banks holding no pages of the buffer are omitted.
std::vector<DramBankExtent> get_dram_bank_extents(const distributed::MeshBuffer& mesh_buffer);

enum class ExportBounds {
    // The dma-buf covers exactly the requested range. The range must start and end on host page boundaries.
    Exact,
    // The dma-buf is widened to the enclosing host page boundaries. Bytes outside
    // [data_offset(), data_offset() + data_size()) belong to other allocations, and the peer must not write them.
    PageCovering,
};

class DmabufExport;

// Returns the largest number of bytes, starting `offset` bytes into `extent`, that a single export can cover.
// Exports are backed by one fixed-size, size-aligned device window, so a range that crosses a window boundary has to be
// split into several exports.
DeviceAddr max_export_size(const distributed::MeshBuffer& mesh_buffer, const DramBankExtent& extent, DeviceAddr offset);

// Exports `size` bytes, starting `offset` bytes into `extent`, as a dma-buf.
// The device at `extent.device_coord` must have its own PCIe link, and `size` must not exceed `max_export_size`.
// The export keeps `mesh_buffer` alive, but explicitly deallocating it while the export exists lets the peer write
// memory that may already belong to another allocation.
DmabufExport export_dram_bank_range(
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    const DramBankExtent& extent,
    DeviceAddr offset,
    DeviceAddr size,
    ExportBounds bounds);

// Blocks until all work enqueued on every command queue of the buffer's MeshDevice has completed.
// Call before a peer starts writing: the buffer's memory may previously have belonged to a buffer that programs still
// in flight are using.
void wait_for_device_idle(const distributed::MeshBuffer& mesh_buffer);

// Call after the peer has reported completion of all its writes into exports of `mesh_buffer`, and before enqueueing
// any program that reads the buffer.
void flush_external_writes(const distributed::MeshBuffer& mesh_buffer);

// Owns a dma-buf file descriptor exported from device memory; closes it on destruction.
class DmabufExport {
public:
    DmabufExport(const DmabufExport&) = delete;
    DmabufExport& operator=(const DmabufExport&) = delete;
    DmabufExport(DmabufExport&& other) noexcept;
    DmabufExport& operator=(DmabufExport&& other) noexcept;
    ~DmabufExport();

    int fd() const { return fd_; }
    // Length of the dma-buf, which is the length a peer registers it with.
    size_t size() const { return size_; }
    // Location of the requested range within the dma-buf.
    size_t data_offset() const { return data_offset_; }
    size_t data_size() const { return data_size_; }

private:
    DmabufExport(
        int fd,
        size_t size,
        size_t data_offset,
        size_t data_size,
        std::shared_ptr<distributed::MeshBuffer> mesh_buffer);

    void reset() noexcept;

    friend DmabufExport export_dram_bank_range(
        const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
        const DramBankExtent& extent,
        DeviceAddr offset,
        DeviceAddr size,
        ExportBounds bounds);

    int fd_ = -1;
    size_t size_ = 0;
    size_t data_offset_ = 0;
    size_t data_size_ = 0;
    std::shared_ptr<distributed::MeshBuffer> mesh_buffer_;
};

}  // namespace tt::tt_metal::experimental::dmabuf_export
