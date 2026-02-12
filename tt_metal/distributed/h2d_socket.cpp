// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/hw/inc/hostdev/socket.h"
#include "tt_metal/llrt/tt_cluster.hpp"
#include <tt-metalium/tt_align.hpp>
#include <umd/device/chip_helpers/tlb_manager.hpp>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>

namespace tt::tt_metal::distributed {

H2DSocket::PinnedBufferInfo H2DSocket::init_bytes_acked_buffer(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinateRangeSet& device_range,
    uint32_t pcie_alignment) {
    // Use mmap to ensure page-aligned allocation that won't share pages with other PinnedMemory objects.
    // This prevents failures when multiple sockets try to pin overlapping page regions to the NOC, since
    // the driver does not allow this.
    size_t page_size = sysconf(_SC_PAGESIZE);  // OS Specified Page Size
    // Allocate a single page for the bytes_acked buffer, since its only 4 bytes.
    void* aligned_ptr = mmap(nullptr, page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    TT_FATAL(aligned_ptr != MAP_FAILED, "Failed to allocate page-aligned memory for bytes_acked buffer.");
    TT_FATAL(
        reinterpret_cast<uintptr_t>(aligned_ptr) % pcie_alignment == 0,
        "System Memory Allocation Error: Bytes_acked buffer must be aligned to the PCIe alignment.");
    std::memset(aligned_ptr, 0, sizeof(uint32_t));
    host_buffer_ = std::shared_ptr<uint32_t[]>(
        static_cast<uint32_t*>(aligned_ptr), [page_size](uint32_t* p) { munmap(p, page_size); });
    tt::tt_metal::HostBuffer bytes_acked_buffer_view(
        tt::stl::Span<uint32_t>(host_buffer_.get(), 1), tt::tt_metal::MemoryPin(host_buffer_));
    pinned_memory_ =
        tt::tt_metal::experimental::PinnedMemory::Create(*mesh_device, device_range, bytes_acked_buffer_view, true);

    const auto& noc_addr = pinned_memory_->get_noc_addr(mesh_device->get_device(recv_core_.device_coord)->id());
    TT_FATAL(noc_addr.has_value(), "Failed to get NOC address for bytes_acked pinned memory.");
    TT_FATAL(
        noc_addr.value().device_id == mesh_device->get_device(recv_core_.device_coord)->id(),
        "Pinned Memory used for H2D sockets must be mapped to the same device as the receiver core. H2D Sockets cannot "
        "communicate with remote devices");
    return PinnedBufferInfo{
        .pcie_xy_enc = noc_addr.value().pcie_xy_enc,
        .addr_lo = static_cast<uint32_t>(noc_addr.value().addr & 0xFFFFFFFFull),
        .addr_hi = static_cast<uint32_t>(noc_addr.value().addr >> 32)};
}

H2DSocket::PinnedBufferInfo H2DSocket::init_host_data_buffer(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinateRangeSet& device_range,
    uint32_t pcie_alignment) {
    // Use mmap to ensure page-aligned allocation that won't share pages with other PinnedMemory objects.
    // This prevents failures when multiple sockets try to pin overlapping page regions to the NOC, since
    // the driver does not allow this.
    uint32_t host_buffer_size_bytes = fifo_size_ + sizeof(uint32_t);
    uint32_t host_buffer_size_words = host_buffer_size_bytes / sizeof(uint32_t);
    size_t page_size = sysconf(_SC_PAGESIZE);  // OS Specified Page Size
    // Round up to page boundary
    size_t alloc_size = align(host_buffer_size_bytes, page_size);
    void* aligned_ptr = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    TT_FATAL(aligned_ptr != MAP_FAILED, "Failed to allocate page-aligned memory for host data buffer.");
    TT_FATAL(
        reinterpret_cast<uintptr_t>(aligned_ptr) % pcie_alignment == 0,
        "System Memory Allocation Error: Host data buffer must be aligned to the PCIe alignment.");
    std::memset(aligned_ptr, 0, host_buffer_size_bytes);
    host_buffer_ = std::shared_ptr<uint32_t[]>(
        static_cast<uint32_t*>(aligned_ptr), [alloc_size](uint32_t* p) { munmap(p, alloc_size); });

    tt::tt_metal::HostBuffer host_buffer_view(
        tt::stl::Span<uint32_t>(host_buffer_.get(), host_buffer_size_words), tt::tt_metal::MemoryPin(host_buffer_));
    pinned_memory_ =
        tt::tt_metal::experimental::PinnedMemory::Create(*mesh_device, device_range, host_buffer_view, true);

    const auto& noc_addr = pinned_memory_->get_noc_addr(mesh_device->get_device(recv_core_.device_coord)->id());
    TT_FATAL(noc_addr.has_value(), "Failed to get NOC address for data pinned memory.");

    return PinnedBufferInfo{
        .pcie_xy_enc = noc_addr.value().pcie_xy_enc,
        .addr_lo = static_cast<uint32_t>(noc_addr.value().addr & 0xFFFFFFFFull),
        .addr_hi = static_cast<uint32_t>(noc_addr.value().addr >> 32)};
}

void H2DSocket::init_config_buffer(const std::shared_ptr<MeshDevice>& mesh_device) {
    uint32_t config_buffer_size = sizeof(receiver_socket_md);
    auto num_cores = 1;
    auto shard_params = ShardSpecBuffer(
        CoreRangeSet(recv_core_.core_coord),
        {1, 1},
        ShardOrientation::ROW_MAJOR,
        {1, 1},
        {static_cast<uint32_t>(num_cores), 1});

    DeviceLocalBufferConfig config_buffer_specs = {
        .page_size = config_buffer_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = std::nullopt,
        .sub_device_id = std::nullopt,
    };

    MeshBufferConfig config_mesh_buffer_specs = ReplicatedBufferConfig{
        .size = config_buffer_size,
    };
    config_buffer_ = MeshBuffer::create(config_mesh_buffer_specs, config_buffer_specs, mesh_device.get());
}

void H2DSocket::init_data_buffer(const std::shared_ptr<MeshDevice>& mesh_device, uint32_t pcie_alignment) {
    auto num_data_cores = mesh_device->num_worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0});
    auto shard_grid = mesh_device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0});

    // Allocate buffer at a PCIe aligned address. This requires extra memory to be allocated.
    auto total_data_buffer_size = num_data_cores * (fifo_size_ + pcie_alignment);

    DeviceLocalBufferConfig data_buffer_specs = {
        .page_size = fifo_size_ + pcie_alignment,
        .buffer_type = buffer_type_,
        .sharding_args = BufferShardingArgs(
            ShardSpecBuffer(shard_grid, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_data_cores, 1}),
            TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = std::nullopt,
        .sub_device_id = std::nullopt,
    };
    MeshBufferConfig data_mesh_buffer_specs = ReplicatedBufferConfig{
        .size = total_data_buffer_size,
    };
    data_buffer_ = MeshBuffer::create(data_mesh_buffer_specs, data_buffer_specs, mesh_device.get());
    aligned_data_buf_start_ = tt::align(data_buffer_->address(), pcie_alignment);
    write_ptr_ = 0;
}

void H2DSocket::write_socket_metadata(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const PinnedBufferInfo& bytes_acked_info,
    const PinnedBufferInfo& data_info) {
    const auto& core_to_core_id = config_buffer_->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id;

    std::vector<receiver_socket_md> config_data(
        config_buffer_->size() / sizeof(receiver_socket_md), receiver_socket_md());

    auto& md = config_data[core_to_core_id.at(recv_core_.core_coord)];
    md.bytes_sent = 0;
    md.bytes_acked = 0;
    md.read_ptr = aligned_data_buf_start_;
    md.fifo_addr = aligned_data_buf_start_;
    md.fifo_total_size = fifo_size_;
    md.is_h2d = 1;
    md.h2d.bytes_acked_addr_lo = bytes_acked_info.addr_lo;
    md.h2d.bytes_acked_addr_hi = bytes_acked_info.addr_hi;
    md.h2d.data_addr_lo = data_info.addr_lo;
    md.h2d.data_addr_hi = data_info.addr_hi;
    md.h2d.pcie_xy_enc = bytes_acked_info.pcie_xy_enc;

    distributed::WriteShard(
        mesh_device->mesh_command_queue(0), config_buffer_, config_data, recv_core_.device_coord, true);
}

void H2DSocket::init_receiver_tlb(const std::shared_ptr<MeshDevice>& mesh_device) {
    const auto& cluster = MetalContext::instance().get_cluster();
    auto recv_device_id = mesh_device->get_device(recv_core_.device_coord)->id();
    auto recv_virtual_core = mesh_device->worker_core_from_logical_core(recv_core_.core_coord);
    receiver_core_tlb_ = cluster.get_driver()
                             ->get_chip(recv_device_id)
                             ->get_tlb_manager()
                             ->get_tlb_window(tt_xy_pair(recv_virtual_core.x, recv_virtual_core.y));
    auto arch = MetalContext::instance().hal().get_arch();
    if (arch == tt::ARCH::BLACKHOLE) {
        // Entire device address space for Blackhole is statically mapped.
        // Safe to use static TLBs without requiring the driver to do a reconfig.
        pcie_writer = [&](void* data, uint32_t num_bytes, uint64_t device_addr) {
            receiver_core_tlb_->write_block(device_addr, data, num_bytes);
        };
    } else if (arch == tt::ARCH::WORMHOLE_B0) {
        // Wormhole B0 may require the driver to do a reconfig of the TLB for each write,
        // since the device address space is not statically mapped.
        pcie_writer = [recv_device_id, recv_virtual_core](void* data, uint32_t num_bytes, uint64_t device_addr) {
            const auto& cluster = MetalContext::instance().get_cluster();
            cluster.write_core(data, num_bytes, tt_cxy_pair(recv_device_id, recv_virtual_core), device_addr);
        };
    } else {
        TT_THROW("Unsupported architecture: {}", arch);
    }
}

H2DSocket::H2DSocket(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoreCoord& recv_core,
    BufferType buffer_type,
    uint32_t fifo_size,
    H2DMode h2d_mode) :
    recv_core_(recv_core),
    buffer_type_(buffer_type),
    fifo_size_(fifo_size),
    pinned_memory_(nullptr),
    h2d_mode_(h2d_mode) {
    MeshCoordinateRangeSet recv_device_range_set;
    recv_device_range_set.merge(MeshCoordinateRange(recv_core_.device_coord));

    const uint32_t pcie_alignment = MetalContext::instance().hal().get_alignment(HalMemType::HOST);
    TT_FATAL(fifo_size_ % pcie_alignment == 0, "FIFO size must be PCIE-aligned.");
    TT_FATAL(buffer_type_ == BufferType::L1, "H2D sockets currently only support data buffers in SRAM.");

    PinnedBufferInfo bytes_acked_info = {};
    PinnedBufferInfo data_info = {};
    if (h2d_mode_ == H2DMode::DEVICE_PULL) {
        // Allocate host data buffer and bytes_acked buffer in the same pinned memory.
        data_info = init_host_data_buffer(mesh_device, recv_device_range_set, pcie_alignment);
        bytes_acked_info = data_info;
        // Bytes acked buffer is located after the data buffer in the same pinned memory.
        auto bytes_acked_addr = (static_cast<uint64_t>(data_info.addr_hi) << 32 | data_info.addr_lo) + fifo_size_;
        bytes_acked_info.addr_hi = static_cast<uint32_t>(bytes_acked_addr >> 32);
        bytes_acked_info.addr_lo = static_cast<uint32_t>(bytes_acked_addr & 0xFFFFFFFFull);
        bytes_acked_ptr_ = host_buffer_.get() + (fifo_size_ / sizeof(uint32_t));
        TT_FATAL(
            bytes_acked_info.pcie_xy_enc == data_info.pcie_xy_enc,
            "Bytes_acked and data pinned memory must be mapped to the same PCIe core.");
    } else {
        // Dedicate a separate pinned memory for the bytes_acked buffer.
        bytes_acked_info = init_bytes_acked_buffer(mesh_device, recv_device_range_set, pcie_alignment);
        bytes_acked_ptr_ = host_buffer_.get();
    }

    init_config_buffer(mesh_device);
    init_data_buffer(mesh_device, pcie_alignment);
    write_socket_metadata(mesh_device, bytes_acked_info, data_info);
    init_receiver_tlb(mesh_device);
}

H2DSocket::~H2DSocket() noexcept {
    // Wait for 1000ms for the device to acknowledge all data over the socket.
    // This may need to be tuned in future, depending on the application and
    // the amount of data being sent.
    // Realistically a hang should not be seen here since most user workloads
    // synchronize with the device before the application exits and destructors are called.
    barrier(1000);
    pinned_memory_.reset();
}

void H2DSocket::reserve_bytes(uint32_t num_bytes) {
    uint32_t bytes_free = fifo_size_ - (bytes_sent_ - bytes_acked_);
    while (bytes_free < num_bytes) {
        tt_driver_atomics::mfence();
        volatile uint32_t bytes_acked_value = bytes_acked_ptr_[0];
        bytes_free = fifo_size_ - (bytes_sent_ - bytes_acked_value);
        bytes_acked_ = bytes_acked_value;
    }
}

void H2DSocket::push_bytes(uint32_t num_bytes) {
    if (write_ptr_ + num_bytes >= fifo_curr_size_) {
        write_ptr_ = write_ptr_ + num_bytes - fifo_curr_size_;
        bytes_sent_ += num_bytes + fifo_size_ - fifo_curr_size_;
    } else {
        write_ptr_ += num_bytes;
        bytes_sent_ += num_bytes;
    }
}

void H2DSocket::notify_receiver() {
    uint32_t bytes_sent_addr = config_buffer_->address() + offsetof(receiver_socket_md, bytes_sent);
    pcie_writer(&bytes_sent_, sizeof(bytes_sent_), bytes_sent_addr);
    tt_driver_atomics::sfence();
}

void H2DSocket::set_page_size(uint32_t page_size) {
    const auto pcie_alignment = MetalContext::instance().hal().get_alignment(HalMemType::HOST);
    TT_FATAL(page_size % pcie_alignment == 0, "Page size must be PCIE-aligned.");
    TT_FATAL(page_size <= fifo_size_, "Page size must be less than or equal to the FIFO size.");

    uint32_t next_fifo_wr_ptr = align(write_ptr_, page_size);
    uint32_t fifo_page_aligned_size = fifo_size_ - (fifo_size_ % page_size);

    if (next_fifo_wr_ptr >= fifo_page_aligned_size) {
        bytes_sent_ += fifo_size_ - next_fifo_wr_ptr;
        next_fifo_wr_ptr = 0;
    }
    write_ptr_ = next_fifo_wr_ptr;
    page_size_ = page_size;
    fifo_curr_size_ = fifo_page_aligned_size;
}

void H2DSocket::barrier(std::optional<uint32_t> timeout_ms) {
    volatile uint32_t bytes_acked_value = bytes_acked_ptr_[0];
    auto start_time = std::chrono::high_resolution_clock::now();
    while (bytes_sent_ - bytes_acked_value != 0) {
        tt_driver_atomics::mfence();
        bytes_acked_value = bytes_acked_ptr_[0];
        if (timeout_ms.has_value()) {
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::high_resolution_clock::now() - start_time)
                                  .count();
            if (elapsed_ms > timeout_ms.value()) {
                TT_THROW(
                    "Timeout waiting for device to send acknowledgement over H2D socket. Bytes sent: {}, Bytes "
                    "acknowledged: {}",
                    bytes_sent_,
                    bytes_acked_value);
            }
        }
    }
}

void H2DSocket::write(void* data, uint32_t num_pages) {
    TT_FATAL(page_size_ > 0, "Page size must be set before writing.");
    uint32_t num_bytes = num_pages * page_size_;
    TT_FATAL(num_bytes <= fifo_curr_size_, "Cannot write more pages than the socket FIFO size.");
    auto data_addr = aligned_data_buf_start_ + write_ptr_;
    this->reserve_bytes(num_bytes);

    if (h2d_mode_ == H2DMode::HOST_PUSH) {
        pcie_writer(data, num_bytes, data_addr);
        tt_driver_atomics::sfence();
    } else {
        uint32_t* data_ptr = host_buffer_.get() + (write_ptr_ / sizeof(uint32_t));
        std::memcpy(data_ptr, data, num_bytes);
    }
    this->push_bytes(num_bytes);
    this->notify_receiver();
}

}  // namespace tt::tt_metal::distributed
