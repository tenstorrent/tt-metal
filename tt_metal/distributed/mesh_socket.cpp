// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/distributed_context.hpp>
#include "tt_metal/hw/inc/socket.h"
#include "tt_metal/llrt/tt_cluster.hpp"
#include <tt-metalium/tt_align.hpp>
#include "tt_metal/distributed/fd_mesh_command_queue.hpp"

using namespace tt::tt_metal::distributed::multihost;

namespace tt::tt_metal::distributed {

namespace {

void point_to_point_barrier(
    const std::vector<Rank>& ranks, const std::shared_ptr<multihost::DistributedContext>& distributed_context) {
    TT_FATAL(ranks.size() == 2, "Point-to-point barrier requires exactly two ranks.");
    TT_FATAL(ranks[0] != ranks[1], "Point-to-Point barrier cannot be used for synchronization within the same rank.");
    TT_FATAL(
        distributed_context->rank() == ranks[0] || distributed_context->rank() == ranks[1],
        "Point-to-Point barrier for ranks {} and {} cannot be called on rank {}.",
        *ranks[0],
        *ranks[1],
        *distributed_context->rank());

    if (distributed_context->rank() == ranks[0]) {
        int sync_msg = 1;
        distributed_context->ssend(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&sync_msg), sizeof(sync_msg)), ranks[1], Tag{0});
    } else {
        int sync_msg = 0;
        distributed_context->recv(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&sync_msg), sizeof(sync_msg)), ranks[0], Tag{0});
        TT_FATAL(sync_msg == 1, "Received unexpected message during point-to-point barrier.");
    }
}

}  // namespace

MeshSocket::MeshSocket(const std::shared_ptr<MeshDevice>& device, const SocketConfig& config) : config_(config) {
    auto context = config.distributed_context ? config.distributed_context : DistributedContext::get_current_world();

    if (!(context->rank() == config.sender_rank || context->rank() == config.receiver_rank)) {
        log_warning(
            LogMetal,
            "Creating a null socket on host rank {} with sender rank {} and receiver rank {}.",
            *context->rank(),
            *config.sender_rank,
            *config.receiver_rank);
        return;
    }

    TT_FATAL(
        config.sender_rank != config.receiver_rank,
        "{} must only be used for communication between different host ranks, not within the same rank.",
        __func__);

    bool is_sender = context->rank() == config.sender_rank;
    if (is_sender) {
        socket_endpoint_type_ = SocketEndpoint::SENDER;
        config_buffer_ = create_socket_config_buffer(device, config, socket_endpoint_type_);
    } else {
        socket_endpoint_type_ = SocketEndpoint::RECEIVER;
        config_buffer_ = create_socket_config_buffer(device, config, socket_endpoint_type_);
        data_buffer_ = create_socket_data_buffer(device, config);
    }
    this->connect_with_peer(context);
}

void MeshSocket::connect_with_peer(const std::shared_ptr<multihost::DistributedContext>& context) {
    auto local_endpoint_desc = generate_local_endpoint_descriptor(*this, context->id());
    SocketPeerDescriptor remote_endpoint_desc;
    // Convention:
    //  - Sender Endpoint sends its descriptor first, then receives the peer's descriptor.
    //  - Receiver Endpoint receives the peer's descriptor first, then sends its own descriptor.
    // Asymmetry ensures that the blocking send/recv do not deadlock.
    if (socket_endpoint_type_ == SocketEndpoint::SENDER) {
        forward_descriptor_to_peer(local_endpoint_desc, socket_endpoint_type_, context);
        remote_endpoint_desc =
            receive_and_verify_descriptor_from_peer(local_endpoint_desc, socket_endpoint_type_, context);
        fabric_node_id_map_ = generate_fabric_node_id_map(config_, local_endpoint_desc, remote_endpoint_desc);
    } else {
        remote_endpoint_desc =
            receive_and_verify_descriptor_from_peer(local_endpoint_desc, socket_endpoint_type_, context);
        forward_descriptor_to_peer(local_endpoint_desc, socket_endpoint_type_, context);
        fabric_node_id_map_ = generate_fabric_node_id_map(config_, remote_endpoint_desc, local_endpoint_desc);
    }
    write_socket_configs(config_buffer_, local_endpoint_desc, remote_endpoint_desc, socket_endpoint_type_);
    point_to_point_barrier({config_.sender_rank, config_.receiver_rank}, context);
}

std::pair<MeshSocket, MeshSocket> MeshSocket::create_socket_pair(
    const std::shared_ptr<MeshDevice>& sender,
    const std::shared_ptr<MeshDevice>& receiver,
    const SocketConfig& config) {
    auto sender_config_buffer = create_socket_config_buffer(sender, config, SocketEndpoint::SENDER);
    auto recv_config_buffer = create_socket_config_buffer(receiver, config, SocketEndpoint::RECEIVER);
    auto socket_data_buffer = create_socket_data_buffer(receiver, config);

    auto sender_socket = MeshSocket(
        nullptr,  // The sender socket does not have a data-buffer allocated
        sender_config_buffer,
        config,
        SocketEndpoint::SENDER);
    auto receiver_socket = MeshSocket(socket_data_buffer, recv_config_buffer, config, SocketEndpoint::RECEIVER);

    auto send_peer_descriptor = generate_local_endpoint_descriptor(sender_socket);
    auto recv_peer_descriptor = generate_local_endpoint_descriptor(receiver_socket);

    write_socket_configs(sender_config_buffer, send_peer_descriptor, recv_peer_descriptor, SocketEndpoint::SENDER);
    write_socket_configs(recv_config_buffer, recv_peer_descriptor, send_peer_descriptor, SocketEndpoint::RECEIVER);

    auto fabric_node_id_map = generate_fabric_node_id_map(config, send_peer_descriptor, recv_peer_descriptor);

    sender_socket.fabric_node_id_map_ = fabric_node_id_map;
    receiver_socket.fabric_node_id_map_ = fabric_node_id_map;

    return {sender_socket, receiver_socket};
}

std::shared_ptr<MeshBuffer> MeshSocket::get_data_buffer() const {
    TT_FATAL(data_buffer_, "Cannot access the data buffer for a sender socket.");
    return data_buffer_;
};

std::shared_ptr<MeshBuffer> MeshSocket::get_config_buffer() const { return config_buffer_; }

const SocketConfig& MeshSocket::get_config() const { return config_; }

tt::tt_fabric::FabricNodeId MeshSocket::get_fabric_node_id(SocketEndpoint endpoint, const MeshCoordinate& coord) const {
    return fabric_node_id_map_[static_cast<std::underlying_type_t<SocketEndpoint>>(endpoint)].at(coord);
}

std::unordered_map<MeshCoordinate, std::unordered_set<CoreCoord>> group_recv_cores(
    const std::vector<MeshCoreCoord>& recv_cores) {
    std::unordered_map<MeshCoordinate, std::unordered_set<CoreCoord>> grouped_cores;
    for (const auto& recv_core : recv_cores) {
        grouped_cores[recv_core.device_coord].insert(recv_core.core_coord);
    }
    return grouped_cores;
}

H2DSocket::H2DSocket(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const std::vector<MeshCoreCoord>& recv_cores,
    BufferType buffer_type,
    uint32_t fifo_size) :
    recv_cores_(recv_cores), buffer_type_(buffer_type), fifo_size_(fifo_size), page_size_(0), pinned_memory_(nullptr) {
    // Allocate memory for the config buffer
    MeshCoordinateRangeSet recv_device_range_set;
    for (const auto& recv_core : recv_cores_) {
        recv_device_range_set.merge(MeshCoordinateRange(recv_core.device_coord));
    }
    bytes_acked_buffer_ = std::make_shared<tt::tt_metal::vector_aligned<uint32_t>>(4, 0);

    tt::tt_metal::HostBuffer host_buffer_view(
        tt::stl::Span<uint32_t>(bytes_acked_buffer_->data(), bytes_acked_buffer_->size()),
        tt::tt_metal::MemoryPin(bytes_acked_buffer_));

    pinned_memory_ =
        tt::tt_metal::experimental::PinnedMemory::Create(*mesh_device, recv_device_range_set, host_buffer_view, true);
    const auto& host_noc_addr =
        pinned_memory_->get_noc_addr(mesh_device->get_device(recv_cores_[0].device_coord)->id());
    TT_FATAL(host_noc_addr.has_value(), "Failed to get NOC address for pinned memory.");

    uint32_t host_addr_lo = static_cast<uint32_t>(host_noc_addr.value().addr & 0xFFFFFFFFull);
    uint32_t host_addr_hi = static_cast<uint32_t>(host_noc_addr.value().addr >> 32);
    uint32_t pcie_xy_enc = host_noc_addr.value().pcie_xy_enc;

    uint32_t config_buffer_size = sizeof(receiver_socket_md);
    std::set<CoreRange> all_cores_set;
    std::unordered_map<MeshCoordinate, std::set<CoreRange>> socket_cores_per_device;

    for (const auto& recv_core : recv_cores_) {
        const auto& socket_device = recv_core.device_coord;
        const auto& socket_core = recv_core.core_coord;
        TT_FATAL(
            socket_cores_per_device[socket_device].insert(socket_core).second,
            "Cannot reuse receiver cores in a single socket.");
        all_cores_set.insert(socket_core);
    }

    auto all_cores = CoreRangeSet(all_cores_set);
    auto num_cores = all_cores_set.size();
    auto total_config_buffer_size = num_cores * config_buffer_size;

    auto shard_params = ShardSpecBuffer(all_cores, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1});

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

    auto sub_device_id = SubDeviceId{0};
    auto num_data_cores = mesh_device->num_worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
    auto shard_grid = mesh_device->worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
    auto total_data_buffer_size = num_data_cores * fifo_size_;

    DeviceLocalBufferConfig data_buffer_specs = {
        .page_size = fifo_size_,
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
    write_ptr_ = data_buffer_->address();

    const auto& core_to_core_id = config_buffer_->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id;

    std::vector<receiver_socket_md> config_data(
        config_buffer_->size() / sizeof(receiver_socket_md), receiver_socket_md());

    const auto& grouped_cores = group_recv_cores(recv_cores_);

    for (const auto& [device_coord, cores_set] : grouped_cores) {
        for (const auto& core_coord : cores_set) {
            uint32_t idx = core_to_core_id.at(core_coord);
            auto& md = config_data[idx];
            md.bytes_sent = 0;
            md.bytes_acked = 0;
            md.read_ptr = data_buffer_->address();
            md.fifo_addr = data_buffer_->address();
            md.fifo_total_size = fifo_size_;
            md.is_h2d = 1;
            md.h2d.bytes_acked_addr_lo = host_addr_lo;
            md.h2d.bytes_acked_addr_hi = host_addr_hi;
            md.h2d.pcie_xy_enc = pcie_xy_enc;
        }
        distributed::WriteShard(mesh_device->mesh_command_queue(0), config_buffer_, config_data, device_coord, true);
    }
}

void H2DSocket::reserve_pages(uint32_t num_pages) {
    TT_FATAL(page_size_ > 0, "Page size must be set before reserving pages.");
    // const auto& cluster = MetalContext::instance().get_cluster();
    uint32_t num_bytes = num_pages * page_size_;
    TT_FATAL(num_bytes <= fifo_size_, "Cannot reserve more pages than the socket FIFO size.");

    for (const auto& recv_core : recv_cores_) {
        uint32_t bytes_free = fifo_size_ - (bytes_sent_ - bytes_acked_[recv_core]);
        while (bytes_free < num_bytes) {
            volatile uint32_t bytes_acked_value = bytes_acked_buffer_->at(0);
            bytes_free = fifo_size_ - (bytes_sent_ - bytes_acked_value);
            bytes_acked_[recv_core] = bytes_acked_value;
        }
    }
}

void H2DSocket::push_pages(uint32_t num_pages) {
    TT_FATAL(page_size_ > 0, "Page size must be set before pushing pages.");
    uint32_t num_bytes = num_pages * page_size_;
    TT_FATAL(num_bytes <= fifo_curr_size_, "Cannot push more pages than the socket FIFO size.");

    if (write_ptr_ + num_bytes >= data_buffer_->address() + fifo_curr_size_) {
        write_ptr_ = write_ptr_ + num_bytes - fifo_curr_size_;
        bytes_sent_ += num_bytes + fifo_size_ - fifo_curr_size_;
    } else {
        write_ptr_ += num_bytes;
        bytes_sent_ += num_bytes;
    }
}

void H2DSocket::notify_receiver() {
    const auto& cluster = MetalContext::instance().get_cluster();
    const auto& mesh_device = config_buffer_->device();
    uint32_t bytes_sent_addr = config_buffer_->address() + offsetof(receiver_socket_md, bytes_sent);
    for (const auto& recv_core : recv_cores_) {
        auto recv_virtual_core = mesh_device->worker_core_from_logical_core(recv_core.core_coord);
        auto recv_device_id = mesh_device->get_device(recv_core.device_coord)->id();
        cluster.write_core(
            &bytes_sent_, sizeof(bytes_sent_), tt_cxy_pair(recv_device_id, recv_virtual_core), bytes_sent_addr);
    }
}

void H2DSocket::set_page_size(uint32_t page_size) {
    const auto pcie_alignment = MetalContext::instance().hal().get_alignment(HalMemType::HOST);
    TT_FATAL(page_size % pcie_alignment == 0, "Page size must be PCIE-aligned.");
    TT_FATAL(page_size <= fifo_size_, "Page size must be less than or equal to the FIFO size.");
    uint32_t fifo_start_addr = data_buffer_->address();
    uint32_t next_fifo_wr_ptr = fifo_start_addr + align(write_ptr_ - fifo_start_addr, page_size);
    uint32_t fifo_page_aligned_size = fifo_size_ - (fifo_size_ % page_size);
    uint32_t fifo_page_aligned_limit = fifo_start_addr + fifo_page_aligned_size;
    if (next_fifo_wr_ptr >= fifo_page_aligned_limit) {
        bytes_sent_ += fifo_start_addr + fifo_size_ - next_fifo_wr_ptr;
        next_fifo_wr_ptr = fifo_start_addr;
    }
    write_ptr_ = next_fifo_wr_ptr;
    page_size_ = page_size;
    fifo_curr_size_ = fifo_page_aligned_size;
}

void H2DSocket::barrier() {
    volatile uint32_t bytes_acked_value = bytes_acked_buffer_->at(0);
    while (bytes_sent_ - bytes_acked_value != 0) {
        bytes_acked_value = bytes_acked_buffer_->at(0);
    }
}

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
    uint32_t fifo_size) :
    data_pinned_memory_(nullptr),
    bytes_sent_pinned_memory_(nullptr),
    sender_core_(sender_core),
    buffer_type_(buffer_type),
    fifo_size_(fifo_size),
    page_size_(0) {
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

    const auto& bytes_sent_noc_addr =
        bytes_sent_pinned_memory_->get_noc_addr(mesh_device->get_device(sender_core_.device_coord)->id());
    const auto& data_noc_addr =
        data_pinned_memory_->get_noc_addr(mesh_device->get_device(sender_core_.device_coord)->id());

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

    auto shard_params =
        ShardSpecBuffer(sender_core_range_set, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1});

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

void D2HSocket::wait_for_pages(uint32_t num_pages) {
    TT_FATAL(page_size_ > 0, "Page size must be set before waiting for pages.");
    uint32_t num_bytes = num_pages * page_size_;
    TT_FATAL(num_bytes <= fifo_curr_size_, "Cannot wait for more pages than the socket FIFO size.");

    if (read_ptr_ + num_bytes >= fifo_curr_size_) {
        num_bytes += fifo_size_ - fifo_curr_size_;
    }

    uint32_t bytes_recv = bytes_sent_ - bytes_acked_;
    while (bytes_recv < num_bytes) {
        volatile uint32_t bytes_sent_value = bytes_sent_buffer_->at(0);
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
    auto& mesh_cq = dynamic_cast<FDMeshCommandQueue&>(mesh_device->mesh_command_queue());
    uint32_t bytes_acked_addr = config_buffer_->address() + sender_size.md_size_bytes;
    auto sender_virtual_core = mesh_device->worker_core_from_logical_core(sender_core_.core_coord);

    mesh_cq.enqueue_write_shard_to_core(
        {sender_core_.device_coord, sender_virtual_core, bytes_acked_addr},
        &bytes_acked_,
        sizeof(bytes_acked_),
        false,
        {},
        false);
}

void D2HSocket::barrier() {
    volatile uint32_t bytes_sent_value = bytes_sent_buffer_->at(0);
    while (bytes_acked_ - bytes_sent_value != 0) {
        bytes_sent_value = bytes_sent_buffer_->at(0);
    }
}

}  // namespace tt::tt_metal::distributed

namespace std {

std::size_t hash<tt::tt_metal::distributed::SocketConnection>::operator()(
    const tt::tt_metal::distributed::SocketConnection& conn) const noexcept {
    return tt::stl::hash::hash_objects_with_default_seed(conn.sender_core, conn.receiver_core);
}

std::size_t hash<tt::tt_metal::distributed::MeshCoreCoord>::operator()(
    const tt::tt_metal::distributed::MeshCoreCoord& coord) const noexcept {
    return tt::stl::hash::hash_objects_with_default_seed(coord.device_coord, coord.core_coord);
}

std::size_t hash<tt::tt_metal::distributed::SocketConfig>::operator()(
    const tt::tt_metal::distributed::SocketConfig& config) const noexcept {
    std::optional<tt::tt_metal::distributed::multihost::Rank> distributed_context_rank = std::nullopt;
    std::optional<tt::tt_metal::distributed::multihost::Size> distributed_context_size = std::nullopt;
    if (config.distributed_context) {
        distributed_context_rank = config.distributed_context->rank();
        distributed_context_size = config.distributed_context->size();
    }
    return tt::stl::hash::hash_objects_with_default_seed(
        config.socket_connection_config,
        config.socket_mem_config,
        config.sender_rank,
        config.receiver_rank,
        distributed_context_rank,
        distributed_context_size);
}

std::size_t hash<tt::tt_metal::distributed::MeshSocket>::operator()(
    const tt::tt_metal::distributed::MeshSocket& socket) const noexcept {
    return tt::stl::hash::hash_objects_with_default_seed(socket.attribute_values());
}

}  // namespace std
