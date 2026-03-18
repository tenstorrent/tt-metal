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
#include <tt-logger/tt-logger.hpp>
#include "impl/dispatch/system_memory_manager.hpp"
#include <umd/device/chip_helpers/tlb_manager.hpp>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <immintrin.h>

namespace tt::tt_metal::distributed {

D2HSocket::PinnedBufferInfo D2HSocket::init_host_buffer(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinateRangeSet& device_range,
    uint32_t pcie_alignment) {
    // Buffer layout: [data_region (fifo_size bytes)][bytes_sent (4 bytes)]
    uint32_t total_buffer_size_bytes = fifo_size_ + sizeof(uint32_t);
    uint32_t total_buffer_size_words = total_buffer_size_bytes / sizeof(uint32_t);
    size_t page_size = sysconf(_SC_PAGESIZE);
    size_t alloc_size = align(total_buffer_size_bytes, page_size);
    void* aligned_ptr = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    TT_FATAL(aligned_ptr != MAP_FAILED, "Failed to allocate page-aligned memory for D2H socket buffer.");
    TT_FATAL(
        reinterpret_cast<uintptr_t>(aligned_ptr) % pcie_alignment == 0,
        "System Memory Allocation Error: D2H socket buffer must be aligned to the PCIe alignment.");
    std::memset(aligned_ptr, 0, total_buffer_size_bytes);
    host_buffer_ = std::shared_ptr<uint32_t[]>(
        static_cast<uint32_t*>(aligned_ptr), [alloc_size](uint32_t* p) { munmap(p, alloc_size); });
    bytes_sent_ptr_ = host_buffer_.get() + (fifo_size_ / sizeof(uint32_t));

    tt::tt_metal::HostBuffer host_buffer_view(
        tt::stl::Span<uint32_t>(host_buffer_.get(), total_buffer_size_words), tt::tt_metal::MemoryPin(host_buffer_));
    pinned_memory_ =
        tt::tt_metal::experimental::PinnedMemory::Create(*mesh_device, device_range, host_buffer_view, true);

    const auto& noc_addr = pinned_memory_->get_noc_addr(mesh_device->get_device(sender_core_.device_coord)->id());
    TT_FATAL(noc_addr.has_value(), "Failed to get NOC address for D2H socket pinned memory.");
    TT_FATAL(
        noc_addr.value().device_id == mesh_device->get_device(sender_core_.device_coord)->id(),
        "Pinned Memory used for D2H sockets must be mapped to the same device as the sender core. D2H Sockets cannot "
        "communicate with remote devices.");

    return PinnedBufferInfo{
        .pcie_xy_enc = noc_addr.value().pcie_xy_enc,
        .addr_lo = static_cast<uint32_t>(noc_addr.value().addr & 0xFFFFFFFFull),
        .addr_hi = static_cast<uint32_t>(noc_addr.value().addr >> 32)};
}

D2HSocket::PinnedBufferInfo D2HSocket::init_host_buffer_hugepage(const std::shared_ptr<MeshDevice>& mesh_device) {
    using_hugepage_ = true;

    auto* device = mesh_device->get_device(sender_core_.device_coord);
    auto device_id = device->id();
    auto& sysmem_mgr = device->sysmem_manager();

    auto [data_host_ptr, data_dev_addr] = sysmem_mgr.allocate_region(fifo_size_);
    hugepage_data_host_ptr_ = static_cast<uint32_t*>(data_host_ptr);
    std::memset(hugepage_data_host_ptr_, 0, fifo_size_);

    const auto& cluster = MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();
    ChipId mmio_device_id = cluster.get_associated_mmio_device(device_id);
    const auto& soc = cluster.get_soc_desc(mmio_device_id);
    const auto& pcie_cores = soc.get_cores(CoreType::PCIE, CoordSystem::NOC0);
    TT_ASSERT(!pcie_cores.empty());
    auto pcie_xy = pcie_cores.front();
    uint32_t pcie_xy_enc = hal.noc_xy_pcie64_encoding(pcie_xy.x, pcie_xy.y);

    log_info(
        tt::LogMetal,
        "D2HSocket: Using hugepage fallback for device {} "
        "(data_dev_addr=0x{:x}, pcie_xy_enc=0x{:x})",
        device_id,
        data_dev_addr,
        pcie_xy_enc);

    return PinnedBufferInfo{.pcie_xy_enc = pcie_xy_enc, .addr_lo = data_dev_addr, .addr_hi = 0};
}

void D2HSocket::init_config_buffer(const std::shared_ptr<MeshDevice>& mesh_device) {
    const SocketSenderSize sender_size;
    uint32_t base_config_size = sender_size.md_size_bytes + sender_size.ack_size_bytes + sender_size.enc_size_bytes;

    // Reserve space for optional L1 data buffer info (address + size), always present for uniform layout
    uint32_t l1_info_size = tt::align(2 * sizeof(uint32_t), sender_size.l1_alignment);
    uint32_t config_buffer_size = base_config_size + l1_info_size;

    auto shard_params = ShardSpecBuffer(
        CoreRangeSet(CoreRange(sender_core_.core_coord)), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

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

void D2HSocket::init_l1_data_buffer(const std::shared_ptr<MeshDevice>& mesh_device, uint32_t requested_size) {
    const uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    l1_data_buffer_size_ = tt::align(requested_size, l1_alignment);

    auto shard_params = ShardSpecBuffer(
        CoreRangeSet(CoreRange(sender_core_.core_coord)), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

    DeviceLocalBufferConfig l1_buffer_specs = {
        .page_size = l1_data_buffer_size_,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = std::nullopt,
        .sub_device_id = std::nullopt,
    };

    MeshBufferConfig l1_mesh_buffer_specs = ReplicatedBufferConfig{
        .size = l1_data_buffer_size_,
    };

    l1_data_buffer_ = MeshBuffer::create(l1_mesh_buffer_specs, l1_buffer_specs, mesh_device.get());
    l1_data_buffer_address_ = l1_data_buffer_->address();
}

void D2HSocket::write_socket_metadata(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const PinnedBufferInfo& data_info,
    const PinnedBufferInfo& bytes_sent_info) {
    const SocketSenderSize sender_size;

    std::vector<uint32_t> config_data(config_buffer_->size() / sizeof(uint32_t), 0);
    config_data[0] = 0;                        // bytes_sent
    config_data[1] = 1;                        // num_downstreams
    config_data[2] = 0;                        // write_ptr (offset from downstream_fifo_addr)
    config_data[3] = bytes_sent_info.addr_lo;  // downstream_bytes_sent_addr
    config_data[4] = data_info.addr_lo;        // downstream_fifo_addr
    config_data[5] = fifo_size_;               // fifo_total_size
    config_data[6] = 1;                        // is_d2h

    uint32_t host_addr_offset = (sender_size.md_size_bytes + sender_size.ack_size_bytes) / sizeof(uint32_t);
    config_data[host_addr_offset] = bytes_sent_info.addr_hi;
    config_data[host_addr_offset + 1] = data_info.addr_hi;
    config_data[host_addr_offset + 2] = data_info.pcie_xy_enc;

    // L1 data buffer info follows the downstream encoding (always present, 0 if unused)
    uint32_t l1_info_offset =
        (sender_size.md_size_bytes + sender_size.ack_size_bytes + sender_size.enc_size_bytes) / sizeof(uint32_t);
    config_data[l1_info_offset] = l1_data_buffer_address_;
    config_data[l1_info_offset + 1] = l1_data_buffer_size_;

    IDevice* device = mesh_device->get_device(sender_core_.device_coord);
    tt::tt_metal::detail::WriteToDeviceL1(
        device, sender_core_.core_coord, config_buffer_->address(), config_data, CoreType::WORKER);
}

void D2HSocket::init_sender_tlb(const std::shared_ptr<MeshDevice>& mesh_device) {
    const auto& cluster = MetalContext::instance().get_cluster();
    auto sender_device_id = mesh_device->get_device(sender_core_.device_coord)->id();
    auto sender_virtual_core = mesh_device->worker_core_from_logical_core(sender_core_.core_coord);

    sender_core_tlb_ = cluster.get_driver()
                           ->get_chip(sender_device_id)
                           ->get_tlb_manager()
                           ->get_tlb_window(tt_xy_pair(sender_virtual_core.x, sender_virtual_core.y));

    auto arch = MetalContext::instance().hal().get_arch();
    if (arch == tt::ARCH::BLACKHOLE) {
        pcie_writer_ = [this](void* data, uint32_t num_bytes, uint64_t device_addr) {
            sender_core_tlb_->write_block(device_addr, data, num_bytes);
        };
    } else if (arch == tt::ARCH::WORMHOLE_B0) {
        pcie_writer_ = [sender_device_id, sender_virtual_core](void* data, uint32_t num_bytes, uint64_t device_addr) {
            const auto& cluster = MetalContext::instance().get_cluster();
            cluster.write_core(data, num_bytes, tt_cxy_pair(sender_device_id, sender_virtual_core), device_addr);
        };
    } else {
        TT_THROW("Unsupported architecture: {}", arch);
    }
}

D2HSocket::D2HSocket(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoreCoord& sender_core,
    uint32_t fifo_size,
    uint32_t l1_data_buffer_size) :
    sender_core_(sender_core), fifo_size_(fifo_size) {
    MeshCoordinateRangeSet sender_device_range_set;
    sender_device_range_set.merge(MeshCoordinateRange(sender_core_.device_coord));

    const auto& cluster = MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();
    const uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    TT_FATAL(fifo_size_ % pcie_alignment == 0, "FIFO size must be PCIe-aligned.");

    bool can_use_pinned_memory = cluster.is_iommu_enabled() || hal.get_supports_64_bit_pcie_addressing();

    PinnedBufferInfo data_info;
    PinnedBufferInfo bytes_sent_info;

    if (can_use_pinned_memory) {
        data_info = init_host_buffer(mesh_device, sender_device_range_set, pcie_alignment);
        uint64_t bytes_sent_addr = (static_cast<uint64_t>(data_info.addr_hi) << 32 | data_info.addr_lo) + fifo_size_;
        bytes_sent_info = data_info;
        bytes_sent_info.addr_lo = static_cast<uint32_t>(bytes_sent_addr & 0xFFFFFFFFull);
        bytes_sent_info.addr_hi = static_cast<uint32_t>(bytes_sent_addr >> 32);
    } else {
        data_info = init_host_buffer_hugepage(mesh_device);

        auto* device = mesh_device->get_device(sender_core_.device_coord);
        auto& sysmem_mgr = device->sysmem_manager();
        auto [bs_host_ptr, bs_dev_addr] = sysmem_mgr.allocate_region(sizeof(uint32_t));
        hugepage_bytes_sent_host_ptr_ = static_cast<volatile uint32_t*>(bs_host_ptr);
        *const_cast<uint32_t*>(hugepage_bytes_sent_host_ptr_) = 0;

        bytes_sent_info = data_info;
        bytes_sent_info.addr_lo = bs_dev_addr;
        bytes_sent_info.addr_hi = 0;
    }

    if (l1_data_buffer_size > 0) {
        init_l1_data_buffer(mesh_device, l1_data_buffer_size);
    }

    init_config_buffer(mesh_device);
    write_socket_metadata(mesh_device, data_info, bytes_sent_info);
    init_sender_tlb(mesh_device);
}

D2HSocket::~D2HSocket() noexcept {
    barrier(1000);
    if (!using_hugepage_) {
        pinned_memory_.reset();
    }
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
            volatile uint32_t bytes_sent_value = using_hugepage_ ? *hugepage_bytes_sent_host_ptr_ : bytes_sent_ptr_[0];
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

void D2HSocket::wait_for_bytes(uint32_t num_bytes) {
    if (read_ptr_ + num_bytes >= fifo_curr_size_) {
        num_bytes += fifo_size_ - fifo_curr_size_;
    }
    uint32_t bytes_recv = bytes_sent_ - bytes_acked_;
    while (bytes_recv < num_bytes) {
        if (using_hugepage_) {
            _mm_clflush(const_cast<void*>(reinterpret_cast<const volatile void*>(hugepage_bytes_sent_host_ptr_)));
            _mm_lfence();
            uint32_t bytes_sent_value = *hugepage_bytes_sent_host_ptr_;
            bytes_recv = bytes_sent_value - bytes_acked_;
            bytes_sent_ = bytes_sent_value;
        } else {
            tt_driver_atomics::mfence();
            volatile uint32_t bytes_sent_value = bytes_sent_ptr_[0];
            bytes_recv = bytes_sent_value - bytes_acked_;
            bytes_sent_ = bytes_sent_value;
        }
    }
}

void D2HSocket::pop_bytes(uint32_t num_bytes) {
    if (read_ptr_ + num_bytes >= fifo_curr_size_) {
        read_ptr_ = read_ptr_ + num_bytes - fifo_curr_size_;
        bytes_acked_ += num_bytes + fifo_size_ - fifo_curr_size_;
    } else {
        read_ptr_ += num_bytes;
        bytes_acked_ += num_bytes;
    }
}

void D2HSocket::notify_sender() {
    const SocketSenderSize sender_size;
    uint32_t bytes_acked_addr = config_buffer_->address() + sender_size.md_size_bytes;
    pcie_writer_(&bytes_acked_, sizeof(bytes_acked_), bytes_acked_addr);
    tt_driver_atomics::sfence();
}

void D2HSocket::barrier(std::optional<uint32_t> timeout_ms) {
    auto read_bytes_sent = [this]() -> uint32_t {
        if (using_hugepage_) {
            _mm_clflush(const_cast<void*>(reinterpret_cast<const volatile void*>(hugepage_bytes_sent_host_ptr_)));
            _mm_lfence();
            return *hugepage_bytes_sent_host_ptr_;
        } else {
            tt_driver_atomics::mfence();
            return bytes_sent_ptr_[0];
        }
    };

    volatile uint32_t bytes_sent_value = read_bytes_sent();
    auto start_time = std::chrono::high_resolution_clock::now();
    while (bytes_acked_ - bytes_sent_value != 0) {
        bytes_sent_value = read_bytes_sent();
        if (timeout_ms.has_value()) {
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::high_resolution_clock::now() - start_time)
                                  .count();
            if (elapsed_ms > timeout_ms.value()) {
                TT_THROW(
                    "Timeout waiting for host to acknowledge data over D2H socket. Bytes sent: {}, Bytes "
                    "acknowledged: {}. Barrier was potentially issued on host before all required reads were "
                    "completed.",
                    bytes_sent_,
                    bytes_sent_value);
            }
        }
    }
}

void D2HSocket::read(void* data, uint32_t num_pages, bool notify_sender) {
    TT_FATAL(page_size_ > 0, "Page size must be set before reading.");
    uint32_t num_bytes = num_pages * page_size_;
    TT_FATAL(num_bytes <= fifo_curr_size_, "Cannot read more pages than the socket FIFO size.");
    this->wait_for_bytes(num_bytes);
    uint32_t* src = using_hugepage_ ? hugepage_data_host_ptr_ + (read_ptr_ / sizeof(uint32_t))
                                    : host_buffer_.get() + (read_ptr_ / sizeof(uint32_t));
    if (using_hugepage_) {
        for (uint32_t i = 0; i < num_bytes; i += 64) {
            _mm_clflush(reinterpret_cast<char*>(src) + i);
        }
        _mm_lfence();
    }
    std::memcpy(data, src, num_bytes);
    this->pop_bytes(num_bytes);

    if (notify_sender) {
        this->notify_sender();
    }
}

uint32_t D2HSocket::pages_available() {
    TT_FATAL(page_size_ > 0, "Page size must be set before checking available pages.");
    uint32_t bytes_sent_value;
    if (using_hugepage_) {
        _mm_clflush(const_cast<void*>(reinterpret_cast<const volatile void*>(hugepage_bytes_sent_host_ptr_)));
        _mm_lfence();
        bytes_sent_value = *hugepage_bytes_sent_host_ptr_;
    } else {
        tt_driver_atomics::mfence();
        bytes_sent_value = bytes_sent_ptr_[0];
    }
    bytes_sent_ = bytes_sent_value;
    uint32_t bytes_recv = bytes_sent_value - bytes_acked_;
    return bytes_recv / page_size_;
}

std::vector<MeshCoreCoord> D2HSocket::get_active_cores() const { return {sender_core_}; }

MeshDevice* D2HSocket::get_mesh_device() const { return config_buffer_->device(); }

}  // namespace tt::tt_metal::distributed
