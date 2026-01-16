// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/hw/inc/hostdev/socket.h"
#include "tt_metal/llrt/tt_cluster.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <immintrin.h>  // For _mm_clflush

namespace tt::tt_metal::distributed {

struct SocketSenderSize_ {
    const uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    const uint32_t md_size_bytes = tt::align(sizeof(sender_socket_md), l1_alignment);
    const uint32_t ack_size_bytes = tt::align(sizeof(uint32_t), l1_alignment);
    const uint32_t enc_size_bytes = tt::align(sizeof(sender_downstream_encoding), l1_alignment);
};

D2HSocket::D2HSocket(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoreCoord& sender_core,
    BufferType buffer_type,
    uint32_t fifo_size,
    uint32_t l1_data_buffer_size) :
    data_pinned_memory_(nullptr),
    bytes_sent_pinned_memory_(nullptr),
    sender_core_(sender_core),
    fifo_size_(fifo_size),
    page_size_(0),
    l1_data_buffer_address_(0),
    l1_data_buffer_size_(0) {
    (void)buffer_type;  // Unused for now

    const SocketSenderSize_ sender_size;
    uint32_t config_buffer_size = sender_size.md_size_bytes + sender_size.ack_size_bytes + sender_size.enc_size_bytes;

    MeshCoordinateRangeSet sender_devce_range_set;
    sender_devce_range_set.merge(MeshCoordinateRange(sender_core.device_coord));
    auto sender_core_range_set = CoreRangeSet(CoreRange(sender_core.core_coord));

    data_buffer_ = std::make_shared<tt::tt_metal::vector_aligned<uint32_t>>(fifo_size_ / sizeof(uint32_t), 0);
    bytes_sent_buffer_ = std::make_shared<tt::tt_metal::vector_aligned<uint32_t>>(4, 0);

    tt::tt_metal::HostBuffer data_buffer_view(
        tt::stl::Span<uint32_t>(data_buffer_->data(), data_buffer_->size()), tt::tt_metal::MemoryPin(data_buffer_));
    tt::tt_metal::HostBuffer bytes_sent_buffer_view(
        tt::stl::Span<uint32_t>(bytes_sent_buffer_->data(), bytes_sent_buffer_->size()),
        tt::tt_metal::MemoryPin(bytes_sent_buffer_));

    data_pinned_memory_ =
        tt::tt_metal::experimental::PinnedMemory::Create(*mesh_device, sender_devce_range_set, data_buffer_view, true);

    bytes_sent_pinned_memory_ = tt::tt_metal::experimental::PinnedMemory::Create(
        *mesh_device, sender_devce_range_set, bytes_sent_buffer_view, true);

    auto* device = mesh_device->get_device(sender_core_.device_coord);
    TT_FATAL(device != nullptr, "D2HSocket: Failed to get device for sender_core");
    auto device_id = device->id();

    const auto& bytes_sent_noc_addr = bytes_sent_pinned_memory_->get_noc_addr(device_id);
    const auto& data_noc_addr = data_pinned_memory_->get_noc_addr(device_id);

    uint32_t data_pcie_xy_enc = data_noc_addr.value().pcie_xy_enc;
    uint32_t bytes_sent_pcie_xy_enc = bytes_sent_noc_addr.value().pcie_xy_enc;
    TT_FATAL(
        data_pcie_xy_enc == bytes_sent_pcie_xy_enc,
        "Data and bytes_sent pinned memory must be mapped to the same PCIe core.");

    uint32_t data_addr_lo = static_cast<uint32_t>(data_noc_addr.value().addr & 0xFFFFFFFFull);
    uint32_t data_addr_hi = static_cast<uint32_t>(data_noc_addr.value().addr >> 32);
    uint32_t bytes_sent_addr_lo = static_cast<uint32_t>(bytes_sent_noc_addr.value().addr & 0xFFFFFFFFull);
    uint32_t bytes_sent_addr_hi = static_cast<uint32_t>(bytes_sent_noc_addr.value().addr >> 32);
    read_ptr_ = 0;

    auto num_cores = sender_core_range_set.size();
    auto total_config_buffer_size = num_cores * config_buffer_size;

    // Allocate config buffer via MeshBuffer
    auto shard_params = ShardSpecBuffer(
        sender_core_range_set, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {static_cast<uint32_t>(num_cores), 1});

    DeviceLocalBufferConfig config_buffer_specs = {
        .page_size = config_buffer_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = std::nullopt,
        .sub_device_id = std::nullopt,
    };

    MeshBufferConfig config_mesh_buffer_specs = ReplicatedBufferConfig{
        .size = total_config_buffer_size,
    };
    config_buffer_ = MeshBuffer::create(config_mesh_buffer_specs, config_buffer_specs, mesh_device.get());

    // Allocate L1 data buffer on sender core if requested (same pattern as config buffer)
    if (l1_data_buffer_size > 0) {
        // Get default page size (use 64 bytes as minimum, matching PCIe alignment)
        const uint32_t default_page_size = 64;
        // Round up to smallest multiple of page size
        l1_data_buffer_size_ = ((l1_data_buffer_size + default_page_size - 1) / default_page_size) * default_page_size;

        // ShardOrientation::ROW_MAJOR is a placeholder - with a single sender core, orientation is irrelevant
        auto l1_data_shard_params = ShardSpecBuffer(
            sender_core_range_set, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {static_cast<uint32_t>(num_cores), 1});

        DeviceLocalBufferConfig l1_data_buffer_specs = {
            .page_size = l1_data_buffer_size_,
            .buffer_type = BufferType::L1,
            .sharding_args = BufferShardingArgs(l1_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
            .bottom_up = std::nullopt,
            .sub_device_id = std::nullopt,
        };

        MeshBufferConfig l1_data_mesh_buffer_specs = ReplicatedBufferConfig{
            .size = num_cores * l1_data_buffer_size_,
        };
        l1_data_buffer_ = MeshBuffer::create(l1_data_mesh_buffer_specs, l1_data_buffer_specs, mesh_device.get());
        l1_data_buffer_address_ = l1_data_buffer_->address();
    }

    // Prepare and write config data
    std::vector<uint32_t> config_data(config_buffer_->size() / sizeof(uint32_t), 0);
    config_data[0] = 0;
    config_data[1] = 1;
    config_data[2] = data_addr_lo;        // Lower 32 bits represent the write ptr
    config_data[3] = bytes_sent_addr_lo;  // Lower 32 bits represent the bytes_sent addr
    config_data[4] = data_addr_lo;
    config_data[5] = fifo_size_;
    config_data[6] = 1;

    uint32_t host_addr_offset = (sender_size.md_size_bytes + sender_size.ack_size_bytes) / sizeof(uint32_t);
    config_data[host_addr_offset] = data_pcie_xy_enc;
    config_data[host_addr_offset + 1] = data_addr_hi;
    config_data[host_addr_offset + 2] = bytes_sent_addr_hi;
    // Store L1 data buffer info in config
    config_data[host_addr_offset + 3] = l1_data_buffer_address_;
    config_data[host_addr_offset + 4] = l1_data_buffer_size_;

    distributed::WriteShard(
        mesh_device->mesh_command_queue(0), config_buffer_, config_data, sender_core_.device_coord, true);
}

void D2HSocket::set_page_size(uint32_t page_size) {
    const auto pcie_alignment = MetalContext::instance().hal().get_alignment(HalMemType::HOST);
    TT_FATAL(page_size % pcie_alignment == 0, "Page size must be PCIE-aligned.");
    TT_FATAL(page_size <= fifo_size_, "Page size must be less than or equal to the FIFO size.");

    uint32_t next_fifo_rd_ptr = align(read_ptr_, page_size);
    uint32_t fifo_page_aligned_size = fifo_size_ - (fifo_size_ % page_size);

    if (next_fifo_rd_ptr >= fifo_page_aligned_size) {
        uint32_t bytes_adjustment = fifo_size_ - next_fifo_rd_ptr;
        uint32_t bytes_recv = bytes_sent_ - bytes_acked_;

        while (bytes_recv < bytes_adjustment) {
            volatile uint32_t bytes_sent_value = bytes_sent_buffer_->at(0);
            bytes_recv = bytes_sent_value - bytes_acked_;
            bytes_sent_ = bytes_sent_value;
        }
        bytes_acked_ += bytes_adjustment;
        next_fifo_rd_ptr = 0;
    }
    read_ptr_ = next_fifo_rd_ptr;
    page_size_ = page_size;
    fifo_curr_size_ = fifo_page_aligned_size;
}

uint32_t D2HSocket::pages_available() {
    TT_FATAL(page_size_ > 0, "Page size must be set before checking available pages.");

    // Flush CPU cache line to force re-read from RAM (where PCIe writes land)
    volatile uint32_t* bytes_sent_ptr = const_cast<volatile uint32_t*>(&bytes_sent_buffer_->at(0));
    _mm_clflush(const_cast<void*>(reinterpret_cast<const volatile void*>(bytes_sent_ptr)));
    _mm_lfence();  // Serialize loads after cache line flush

    uint32_t bytes_sent_value = *bytes_sent_ptr;
    bytes_sent_ = bytes_sent_value;
    uint32_t bytes_recv = bytes_sent_value - bytes_acked_;

    return bytes_recv / page_size_;
}

void D2HSocket::wait_for_pages(uint32_t num_pages) {
    TT_FATAL(page_size_ > 0, "Page size must be set before waiting for pages.");
    uint32_t num_bytes = num_pages * page_size_;
    TT_FATAL(num_bytes <= fifo_curr_size_, "Cannot wait for more pages than the socket FIFO size.");

    if (read_ptr_ + num_bytes >= fifo_curr_size_) {
        num_bytes += fifo_size_ - fifo_curr_size_;
    }

    uint32_t bytes_recv = bytes_sent_ - bytes_acked_;
    volatile uint32_t* bytes_sent_ptr = const_cast<volatile uint32_t*>(&bytes_sent_buffer_->at(0));
    while (bytes_recv < num_bytes) {
        // Flush CPU cache line to force re-read from RAM (where PCIe writes land)
        _mm_clflush(const_cast<void*>(reinterpret_cast<const volatile void*>(bytes_sent_ptr)));
        _mm_lfence();  // Serialize loads after cache line flush
        uint32_t bytes_sent_value = *bytes_sent_ptr;
        bytes_recv = bytes_sent_value - bytes_acked_;
        bytes_sent_ = bytes_sent_value;
    }
}

void D2HSocket::pop_pages(uint32_t num_pages) {
    TT_FATAL(page_size_ > 0, "Page size must be set before popping pages.");
    uint32_t num_bytes = num_pages * page_size_;
    TT_FATAL(num_bytes <= fifo_curr_size_, "Cannot pop more pages than the socket FIFO size.");

    if (read_ptr_ + num_bytes >= fifo_curr_size_) {
        read_ptr_ = read_ptr_ + num_bytes - fifo_curr_size_;
        bytes_acked_ += num_bytes + fifo_size_ - fifo_curr_size_;
    } else {
        read_ptr_ += num_bytes;
        bytes_acked_ += num_bytes;
    }
}

void D2HSocket::notify_sender() {
    const SocketSenderSize_ sender_size;
    const auto& mesh_device = config_buffer_->device();
    uint32_t bytes_acked_addr = config_buffer_->address() + sender_size.md_size_bytes;

    // Use slow dispatch to write bytes_acked to sender core
    // This avoids blocking during trace capture (fast dispatch writes are not allowed during trace)
    IDevice* device = mesh_device->get_device(sender_core_.device_coord);
    std::vector<uint32_t> ack_data = {bytes_acked_};
    tt::tt_metal::detail::WriteToDeviceL1(
        device, sender_core_.core_coord, bytes_acked_addr, ack_data, CoreType::WORKER);
}

void D2HSocket::barrier() {
    volatile uint32_t bytes_sent_value = bytes_sent_buffer_->at(0);
    while (bytes_acked_ - bytes_sent_value != 0) {
        bytes_sent_value = bytes_sent_buffer_->at(0);
    }
}

}  // namespace tt::tt_metal::distributed
