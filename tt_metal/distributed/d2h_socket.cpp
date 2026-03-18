// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include "tt_metal/distributed/named_shm.hpp"
#include "tt_metal/distributed/socket_descriptor.hpp"
#include "tt_metal/distributed/pcie_core_writer.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/hw/inc/hostdev/socket.h"
#include "tt_metal/llrt/tt_cluster.hpp"
#include <tt-metalium/tt_align.hpp>
#include <umd/device/chip_helpers/tlb_manager.hpp>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

namespace tt::tt_metal::distributed {

static std::string generate_d2h_shm_name(const std::string& prefix) {
    static std::atomic<uint32_t> counter{0};
    return fmt::format("/tt_{}_{}_{}", prefix, getpid(), counter.fetch_add(1));
}

D2HSocket::PinnedBufferInfo D2HSocket::init_host_buffer(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinateRangeSet& device_range,
    uint32_t pcie_alignment,
    const std::string& shm_name) {
    // Buffer layout: [data_region (fifo_size bytes)][bytes_sent (4 bytes)]
    uint32_t total_buffer_size_bytes = fifo_size_ + sizeof(uint32_t);
    uint32_t total_buffer_size_words = total_buffer_size_bytes / sizeof(uint32_t);
    size_t page_size = sysconf(_SC_PAGESIZE);
    size_t alloc_size = align(total_buffer_size_bytes, page_size);

    shm_ = std::make_unique<NamedShm>(NamedShm::create(shm_name, alloc_size));
    void* aligned_ptr = shm_->ptr();
    TT_FATAL(
        reinterpret_cast<uintptr_t>(aligned_ptr) % pcie_alignment == 0,
        "System Memory Allocation Error: D2H socket buffer must be aligned to the PCIe alignment.");
    // NamedShm::create zero-initializes the region; no explicit memset needed.
    // No-op deleter: NamedShm owns the memory.
    host_buffer_ = std::shared_ptr<uint32_t[]>(static_cast<uint32_t*>(aligned_ptr), [](uint32_t*) {});
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

void D2HSocket::init_config_buffer(const std::shared_ptr<MeshDevice>& mesh_device) {
    const SocketSenderSize sender_size;
    uint32_t config_buffer_size = sender_size.md_size_bytes + sender_size.ack_size_bytes + sender_size.enc_size_bytes;

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

    distributed::WriteShard(
        mesh_device->mesh_command_queue(0), config_buffer_, config_data, sender_core_.device_coord, true);
}

void D2HSocket::init_sender_tlb(const std::shared_ptr<MeshDevice>& mesh_device, std::optional<uint32_t> device_id) {
    TT_FATAL(mesh_device || device_id.has_value(), "Either mesh_device or device_id must be provided.");

    uint32_t sender_device_id;
    CoreCoord sender_virtual_core;

    const auto& cluster = MetalContext::instance().get_cluster();

    if (mesh_device) {
        sender_device_id = mesh_device->get_device(sender_core_.device_coord)->id();
        sender_virtual_core = mesh_device->worker_core_from_logical_core(sender_core_.core_coord);
        sender_core_tlb_ = cluster.get_driver()
                               ->get_chip(sender_device_id)
                               ->get_tlb_manager()
                               ->get_tlb_window(tt_xy_pair(sender_virtual_core.x, sender_virtual_core.y));
    } else {
        sender_device_id = device_id.value();
        sender_virtual_core = cluster.get_virtual_coordinate_from_logical_coordinates(
            sender_device_id, sender_core_.core_coord, CoreType::TENSIX);
    }

    auto arch = MetalContext::instance().hal().get_arch();
    if (arch == tt::ARCH::BLACKHOLE && mesh_device) {
        pcie_writer_ = [this](void* data, uint32_t num_bytes, uint64_t device_addr) {
            sender_core_tlb_->write_block(device_addr, data, num_bytes);
        };
    } else {
        pcie_writer_ = [sender_device_id, sender_virtual_core](void* data, uint32_t num_bytes, uint64_t device_addr) {
            const auto& cluster = MetalContext::instance().get_cluster();
            cluster.write_core(data, num_bytes, tt_cxy_pair(sender_device_id, sender_virtual_core), device_addr);
        };
    }
}

D2HSocket::D2HSocket(
    const std::shared_ptr<MeshDevice>& mesh_device, const MeshCoreCoord& sender_core, uint32_t fifo_size) :
    sender_core_(sender_core), fifo_size_(fifo_size), mesh_device_(mesh_device.get()), is_owner_(true) {
    MeshCoordinateRangeSet sender_device_range_set;
    sender_device_range_set.merge(MeshCoordinateRange(sender_core_.device_coord));

    pcie_alignment_ = MetalContext::instance().hal().get_alignment(HalMemType::HOST);
    const uint32_t pcie_alignment = pcie_alignment_;
    TT_FATAL(fifo_size_ % pcie_alignment == 0, "FIFO size must be PCIe-aligned.");

    std::string shm_name = generate_d2h_shm_name("d2h");
    PinnedBufferInfo data_info = init_host_buffer(mesh_device, sender_device_range_set, pcie_alignment, shm_name);

    // bytes_sent is located at the end of the data buffer in the same pinned memory
    PinnedBufferInfo bytes_sent_info = data_info;
    uint64_t bytes_sent_addr = (static_cast<uint64_t>(data_info.addr_hi) << 32 | data_info.addr_lo) + fifo_size_;
    bytes_sent_info.addr_lo = static_cast<uint32_t>(bytes_sent_addr & 0xFFFFFFFFull);
    bytes_sent_info.addr_hi = static_cast<uint32_t>(bytes_sent_addr >> 32);

    init_config_buffer(mesh_device);
    write_socket_metadata(mesh_device, data_info, bytes_sent_info);
    init_sender_tlb(mesh_device);

    config_buffer_address_ = config_buffer_->address();
    const SocketSenderSize sender_size;
    bytes_acked_device_offset_ = sender_size.md_size_bytes;
}

D2HSocket::~D2HSocket() noexcept {
    if (!exported_) {
        barrier(1000);
    }
    if (is_owner_) {
        pinned_memory_.reset();
        if (shm_) {
            shm_->unlink();
        }
        if (!descriptor_path_.empty()) {
            std::remove(descriptor_path_.c_str());
        }
    }
}

void D2HSocket::set_page_size(uint32_t page_size) {
    TT_FATAL(pcie_alignment_ > 0, "PCIe alignment not initialized.");
    TT_FATAL(page_size % pcie_alignment_ == 0, "Page size must be PCIE-aligned.");
    TT_FATAL(page_size <= fifo_size_, "Page size must be less than or equal to the FIFO size.");

    uint32_t next_fifo_rd_ptr = align(read_ptr_, page_size);
    uint32_t fifo_page_aligned_size = fifo_size_ - (fifo_size_ % page_size);

    if (next_fifo_rd_ptr >= fifo_page_aligned_size) {
        uint32_t bytes_adjustment = fifo_size_ - next_fifo_rd_ptr;
        uint32_t bytes_recv = bytes_sent_ - bytes_acked_;

        while (bytes_recv < bytes_adjustment) {
            tt_driver_atomics::mfence();
            volatile uint32_t bytes_sent_value = bytes_sent_ptr_[0];
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
        tt_driver_atomics::mfence();
        volatile uint32_t bytes_sent_value = bytes_sent_ptr_[0];
        bytes_recv = bytes_sent_value - bytes_acked_;
        bytes_sent_ = bytes_sent_value;
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
    uint32_t bytes_acked_addr = config_buffer_address_ + bytes_acked_device_offset_;
    pcie_writer_(&bytes_acked_, sizeof(bytes_acked_), bytes_acked_addr);
    tt_driver_atomics::sfence();
}

void D2HSocket::barrier(std::optional<uint32_t> timeout_ms) {
    volatile uint32_t bytes_sent_value = bytes_sent_ptr_[0];
    auto start_time = std::chrono::high_resolution_clock::now();
    while (bytes_acked_ - bytes_sent_value != 0) {
        tt_driver_atomics::mfence();
        bytes_sent_value = bytes_sent_ptr_[0];
        if (timeout_ms.has_value()) {
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::high_resolution_clock::now() - start_time)
                                  .count();
            if (elapsed_ms > timeout_ms.value()) {
                // In single threaded environments, this will happen if the barrier is called
                // before the data buffer in sysmem is cleared.
                // In multi-threaded environments, this will happen if the reader thread is hung.
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
    auto* socket_data_ptr = host_buffer_.get() + (read_ptr_ / sizeof(uint32_t));
    this->wait_for_bytes(num_bytes);
    std::memcpy(data, socket_data_ptr, num_bytes);
    this->pop_bytes(num_bytes);

    if (notify_sender) {
        this->notify_sender();
    }
}

std::vector<MeshCoreCoord> D2HSocket::get_active_cores() const { return {sender_core_}; }

MeshDevice* D2HSocket::get_mesh_device() const { return mesh_device_; }

std::string D2HSocket::export_descriptor(const std::string& socket_id) {
    TT_FATAL(is_owner_, "Only the owner process can export a socket descriptor.");
    TT_FATAL(shm_ && shm_->is_open(), "Cannot export descriptor: shared memory is not initialized.");

    auto device_id = mesh_device_->get_device(sender_core_.device_coord)->id();

    SocketDescriptor desc;
    desc.socket_type = "d2h";
    desc.shm_name = shm_->name();
    desc.shm_size = shm_->size();
    desc.data_offset = 0;
    desc.bytes_sent_offset = fifo_size_;
    desc.bytes_acked_offset = 0;
    desc.fifo_size = fifo_size_;
    desc.h2d_mode = 0;
    desc.config_buffer_address = config_buffer_address_;
    desc.aligned_data_buf_start = 0;
    desc.device_id = static_cast<uint32_t>(device_id);
    desc.core_x = sender_core_.core_coord.x;
    desc.core_y = sender_core_.core_coord.y;

    auto virtual_core = mesh_device_->worker_core_from_logical_core(sender_core_.core_coord);
    desc.virtual_core_x = virtual_core.x;
    desc.virtual_core_y = virtual_core.y;
    desc.pcie_alignment = MetalContext::instance().hal().get_alignment(HalMemType::HOST);
    desc.bytes_acked_device_offset = bytes_acked_device_offset_;

    descriptor_path_ = fmt::format("/dev/shm/tt_d2h_{}.json", socket_id);
    desc.write_to_file(descriptor_path_);
    exported_ = true;
    return descriptor_path_;
}

std::unique_ptr<D2HSocket> D2HSocket::connect(const std::string& socket_id, std::optional<uint32_t> timeout_ms) {
    auto descriptor_path = fmt::format("/dev/shm/tt_d2h_{}.json", socket_id);
    auto start_time = std::chrono::high_resolution_clock::now();
    while (!std::filesystem::exists(descriptor_path)) {
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::high_resolution_clock::now() - start_time)
                              .count();
        if (elapsed_ms > timeout_ms.value_or(10000)) {
            TT_THROW("Timeout waiting for descriptor file to be created: {}", descriptor_path);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    auto desc = SocketDescriptor::read_from_file(descriptor_path);
    TT_FATAL(desc.socket_type == "d2h", "Descriptor type mismatch: expected 'd2h', got '{}'", desc.socket_type);

    auto socket = std::unique_ptr<D2HSocket>(new D2HSocket());
    socket->is_owner_ = false;
    socket->fifo_size_ = desc.fifo_size;
    socket->config_buffer_address_ = desc.config_buffer_address;
    socket->sender_core_ = MeshCoreCoord(MeshCoordinate(0, 0), CoreCoord(desc.core_x, desc.core_y));
    socket->pcie_alignment_ = desc.pcie_alignment;
    socket->bytes_acked_device_offset_ = desc.bytes_acked_device_offset;

    socket->shm_ = std::make_unique<NamedShm>(NamedShm::open(desc.shm_name, desc.shm_size));
    socket->host_buffer_ = std::shared_ptr<uint32_t[]>(static_cast<uint32_t*>(socket->shm_->ptr()), [](uint32_t*) {});
    socket->bytes_sent_ptr_ = static_cast<uint32_t*>(socket->shm_->ptr()) + (desc.bytes_sent_offset / sizeof(uint32_t));

    socket->pcie_writer_instance_ =
        std::make_unique<PCIeCoreWriter>(desc.device_id, desc.virtual_core_x, desc.virtual_core_y);
    socket->pcie_writer_ = socket->pcie_writer_instance_->get_pcie_writer();

    return socket;
}

}  // namespace tt::tt_metal::distributed
