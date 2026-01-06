// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include <algorithm>

#include "tests/tt_metal/multihost/fabric_tests/socket_send_recv_utils.hpp"
#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_fabric::fabric_router_tests::multihost::multihost_utils {

std::string get_system_config_name(SystemConfig system_config) {
    switch (system_config) {
        case SystemConfig::SPLIT_T3K: return "SplitT3K";
        case SystemConfig::DUAL_T3K: return "DualT3K";
        case SystemConfig::NANO_EXABOX: return "NanoExabox";
        case SystemConfig::EXABOX: return "Exabox";
        case SystemConfig::SPLIT_GALAXY: return "SplitGalaxy";
        default: return "Unknown";
    }
}

std::string get_test_variant_name(TestVariant variant) {
    switch (variant) {
        case TestVariant::SINGLE_CONN_BWD: return "MultiMeshSingleConnectionBwd";
        case TestVariant::SINGLE_CONN_FWD: return "MultiMeshSingleConnectionFwd";
        case TestVariant::MULTI_CONN_FWD: return "MultiMeshMultiConnectionFwd";
        case TestVariant::MULTI_CONN_BIDIR: return "MultiConnectionBidirectional";
        default: return "Unknown";
    }
}

uint32_t sync_seed_across_ranks(tt_fabric::MeshId sender_mesh_id, tt_fabric::MeshId recv_mesh_id) {
    using namespace tt::tt_metal::distributed::multihost;
    using namespace tt::tt_metal::distributed;
    uint32_t seed;
    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    std::unordered_map<Rank, Rank> rank_translation_table;
    for (int i = 0; i < *distributed_context->size(); i++) {
        rank_translation_table[Rank{i}] = Rank{i};
    }
    std::vector<Rank> sender_ranks = get_ranks_for_mesh_id(sender_mesh_id, rank_translation_table);
    std::vector<Rank> recv_ranks = get_ranks_for_mesh_id(recv_mesh_id, rank_translation_table);
    Rank controller_rank = *std::min_element(sender_ranks.begin(), sender_ranks.end());
    if (distributed_context->rank() == controller_rank) {
        seed = std::chrono::steady_clock::now().time_since_epoch().count();
        for (const auto& rank : sender_ranks) {
            if (rank == controller_rank) {
                continue;
            }
            distributed_context->send(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&seed), sizeof(seed)),
                rank,
                tt::tt_metal::distributed::multihost::Tag{0});
        }
        for (const auto& rank : recv_ranks) {
            distributed_context->send(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&seed), sizeof(seed)),
                rank,
                tt::tt_metal::distributed::multihost::Tag{0});
        }
    } else {
        distributed_context->recv(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&seed), sizeof(seed)),
            controller_rank,
            tt::tt_metal::distributed::multihost::Tag{0});
    }
    log_info(tt::LogTest, "Using seed: {}", seed);
    return seed;
}

MeshId get_local_mesh_id() {
    auto local_mesh_bindings = tt::tt_metal::MetalContext::instance().get_control_plane().get_local_mesh_id_bindings();
    TT_FATAL(local_mesh_bindings.size() == 1, "Must only have a single local mesh binding.");
    return local_mesh_bindings[0];
}

bool test_socket_send_recv(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device_,
    tt_metal::distributed::MeshSocket& socket,
    uint32_t data_size,
    uint32_t page_size,
    uint32_t num_txns,
    std::optional<std::mt19937> gen) {
    using namespace tt::tt_metal::distributed::multihost;
    using namespace tt::tt_metal::distributed;
    using namespace tt_metal;

    bool is_data_match = true;

    auto fabric_max_packet_size = tt_fabric::get_tt_fabric_max_payload_size_bytes();
    auto packet_header_size_bytes = tt_fabric::get_tt_fabric_packet_header_size_bytes();

    auto my_mesh_id = get_local_mesh_id();

    if (!gen.has_value()) {
        // Synchronize seed across ranks
        uint32_t seed = sync_seed_across_ranks(
            socket.get_config().sender_mesh_id.value(), socket.get_config().receiver_mesh_id.value());
        gen = std::mt19937(seed);
    }

    std::set<CoreRange> sender_core_range;
    std::set<CoreRange> recv_core_range;
    for (const auto& connection : socket.get_config().socket_connection_config) {
        sender_core_range.insert(CoreRange(connection.sender_core.core_coord, connection.sender_core.core_coord));
        recv_core_range.insert(CoreRange(connection.receiver_core.core_coord, connection.receiver_core.core_coord));
    }
    auto sender_core_range_set = CoreRangeSet(sender_core_range);
    auto recv_core_range_set = CoreRangeSet(recv_core_range);

    std::uniform_int_distribution<uint32_t> dis(0, UINT32_MAX);

    std::vector<uint32_t> src_vec_per_core(data_size / sizeof(uint32_t));
    std::generate(src_vec_per_core.begin(), src_vec_per_core.end(), [&]() { return dis(gen.value()); });
    std::vector<uint32_t> src_vec;
    src_vec.reserve(data_size * sender_core_range_set.num_cores() / sizeof(uint32_t));

    // duplicate data for all cores; this is non-ideal but there is no elegant way to not do this with current APIs
    for (int i = 0; i < sender_core_range_set.num_cores(); i++) {
        src_vec.insert(src_vec.end(), src_vec_per_core.begin(), src_vec_per_core.end());
    }
    const auto reserved_packet_header_CB_index = tt::CB::c_in0;

    for (int i = 0; i < num_txns; i++) {
        if (my_mesh_id == socket.get_config().sender_mesh_id.value()) {
            auto sender_data_shard_params = ShardSpecBuffer(
                sender_core_range, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {sender_core_range_set.num_cores(), 1});

            const DeviceLocalBufferConfig sender_device_local_config{
                .page_size = data_size,
                .buffer_type = BufferType::L1,
                .sharding_args = BufferShardingArgs(sender_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
                .bottom_up = false};
            const ReplicatedBufferConfig buffer_config{.size = sender_core_range_set.num_cores() * data_size};

            auto sender_data_buffer = MeshBuffer::create(buffer_config, sender_device_local_config, mesh_device_.get());
            auto sender_mesh_workload = MeshWorkload();
            std::unordered_set<MeshCoreCoord> mesh_core_coords;

            for (const auto& connection : socket.get_config().socket_connection_config) {
                if (mesh_core_coords.contains(connection.sender_core)) {
                    continue;
                }
                mesh_core_coords.insert(connection.sender_core);
                auto sender_core = connection.sender_core.core_coord;
                WriteShard(
                    mesh_device_->mesh_command_queue(),
                    sender_data_buffer,
                    src_vec,
                    connection.sender_core.device_coord);

                auto sender_fabric_node_id = mesh_device_->get_fabric_node_id(connection.sender_core.device_coord);
                auto recv_fabric_node_id =
                    socket.get_fabric_node_id(SocketEndpoint::RECEIVER, connection.receiver_core.device_coord);

                auto sender_program = CreateProgram();

                auto sender_kernel = CreateKernel(
                    sender_program,
                    "tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_sender.cpp",
                    sender_core,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                        .compile_args =
                            {static_cast<uint32_t>(socket.get_config_buffer()->address()),
                             static_cast<uint32_t>(sender_data_buffer->address()),
                             static_cast<uint32_t>(page_size),
                             static_cast<uint32_t>(data_size)},
                        .defines = {{"FABRIC_MAX_PACKET_SIZE", std::to_string(fabric_max_packet_size)}}});

                tt::tt_metal::CircularBufferConfig sender_cb_reserved_packet_header_config =
                    tt::tt_metal::CircularBufferConfig(
                        2 * packet_header_size_bytes, {{reserved_packet_header_CB_index, tt::DataFormat::UInt32}})
                        .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);

                CreateCircularBuffer(sender_program, sender_core, sender_cb_reserved_packet_header_config);

                std::vector<uint32_t> sender_rtas;
                tt_fabric::append_fabric_connection_rt_args(
                    sender_fabric_node_id, recv_fabric_node_id, 0, sender_program, {sender_core}, sender_rtas);

                tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_core, sender_rtas);
                sender_mesh_workload.add_program(
                    MeshCoordinateRange(connection.sender_core.device_coord), std::move(sender_program));
            }
            // Run workload performing Data Movement over the socket
            EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), sender_mesh_workload, false);
            Finish(mesh_device_->mesh_command_queue());
        } else {
            auto recv_data_shard_params = ShardSpecBuffer(
                recv_core_range, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {recv_core_range_set.num_cores(), 1});

            const DeviceLocalBufferConfig recv_device_local_config{
                .page_size = data_size,
                .buffer_type = BufferType::L1,
                .sharding_args = BufferShardingArgs(recv_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
                .bottom_up = false};

            const ReplicatedBufferConfig buffer_config{.size = recv_core_range_set.num_cores() * data_size};
            auto recv_data_buffer = MeshBuffer::create(buffer_config, recv_device_local_config, mesh_device_.get());

            auto recv_mesh_workload = MeshWorkload();
            for (const auto& connection : socket.get_config().socket_connection_config) {
                auto recv_core = connection.receiver_core.core_coord;
                auto sender_fabric_node_id =
                    socket.get_fabric_node_id(SocketEndpoint::SENDER, connection.sender_core.device_coord);
                auto recv_fabric_node_id = mesh_device_->get_fabric_node_id(connection.receiver_core.device_coord);

                auto recv_program = CreateProgram();

                auto recv_virtual_coord = recv_data_buffer->device()->worker_core_from_logical_core(recv_core);

                KernelHandle recv_kernel = CreateKernel(
                    recv_program,
                    "tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_receiver_worker.cpp",
                    recv_core,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                        .compile_args = {
                            static_cast<uint32_t>(socket.get_config_buffer()->address()),
                            static_cast<uint32_t>(page_size),
                            static_cast<uint32_t>(data_size),
                            static_cast<uint32_t>(recv_virtual_coord.x),
                            static_cast<uint32_t>(recv_virtual_coord.y),
                            static_cast<uint32_t>(recv_data_buffer->address())}});

                std::vector<uint32_t> recv_rtas;
                tt_fabric::append_fabric_connection_rt_args(
                    recv_fabric_node_id, sender_fabric_node_id, 0, recv_program, {recv_core}, recv_rtas);
                tt_metal::SetRuntimeArgs(recv_program, recv_kernel, recv_core, recv_rtas);
                recv_mesh_workload.add_program(
                    MeshCoordinateRange(connection.receiver_core.device_coord), std::move(recv_program));
            }
            // Run receiver workload using the created socket
            EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), recv_mesh_workload, false);
            const auto& core_to_core_id =
                recv_data_buffer->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id;
            for (const auto& connection : socket.get_config().socket_connection_config) {
                std::vector<uint32_t> recv_data_readback;
                if (mesh_device_->is_local(connection.receiver_core.device_coord)) {
                    // Only read back data on devices owned by this host
                    ReadShard(
                        mesh_device_->mesh_command_queue(),
                        recv_data_readback,
                        recv_data_buffer,
                        connection.receiver_core.device_coord);
                    uint32_t idx = core_to_core_id.at(connection.receiver_core.core_coord);
                    std::vector<uint32_t> recv_data_readback_per_core(
                        recv_data_readback.begin() + idx * data_size / sizeof(uint32_t),
                        recv_data_readback.begin() + (idx + 1) * data_size / sizeof(uint32_t));
                    is_data_match &= (src_vec_per_core == recv_data_readback_per_core);
                    EXPECT_EQ(src_vec_per_core, recv_data_readback_per_core);
                }
            }
        }
        // Increment the source vector for the next iteration
        // This is to ensure that the data is different for each transaction
        for (unsigned int& val : src_vec) {
            val++;
        }
        for (unsigned int& val : src_vec_per_core) {
            val++;
        }
    }
    return is_data_match;
}

std::vector<tt::tt_fabric::MeshId> get_neighbor_mesh_ids(SystemConfig system_config) {
    std::vector<tt::tt_fabric::MeshId> recv_mesh_ids;

    if (system_config == SystemConfig::NANO_EXABOX || system_config == SystemConfig::EXABOX) {
        // Exabox and Nano-Exabox currently have 5 hosts. Sender ranks assignment is customized for a particular Rank File.
        recv_mesh_ids = {
            tt::tt_fabric::MeshId{0}, tt::tt_fabric::MeshId{2}, tt::tt_fabric::MeshId{3}, tt::tt_fabric::MeshId{4}};
    } else if (
        system_config == SystemConfig::SPLIT_T3K || system_config == SystemConfig::DUAL_T3K ||
        system_config == SystemConfig::SPLIT_GALAXY) {
        // Only a single recv node is needed for the dual host configurations.
        recv_mesh_ids = {tt::tt_fabric::MeshId{0}};
    } else {
        TT_THROW("Unsupported system configuration for multi-mesh single connection test.");
    }
    return recv_mesh_ids;
}

void test_multi_mesh_single_conn_bwd(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t socket_fifo_size,
    uint32_t socket_page_size,
    uint32_t data_size,
    SystemConfig system_config) {
    using namespace tt::tt_metal::distributed::multihost;
    using namespace tt::tt_metal::distributed;
    using namespace tt_metal;

    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();

    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv_logical_coord = CoreCoord(0, 0);

    SocketConnection socket_connection(
        MeshCoreCoord(MeshCoordinate(0, 0), sender_logical_coord),
        MeshCoreCoord(MeshCoordinate(0, 0), recv_logical_coord));

    SocketMemoryConfig socket_mem_config(tt_metal::BufferType::L1, socket_fifo_size);

    auto local_mesh_id = get_local_mesh_id();
    constexpr tt::tt_fabric::MeshId sender_mesh_id = tt::tt_fabric::MeshId{1};
    constexpr uint32_t num_iterations = 50;

    if (local_mesh_id == sender_mesh_id) {
        std::unordered_map<tt::tt_fabric::MeshId, MeshSocket> sockets;
        std::vector<tt::tt_fabric::MeshId> recv_mesh_ids = get_neighbor_mesh_ids(system_config);

        for (const auto& recv_mesh_id : recv_mesh_ids) {
            SocketConfig socket_config =
                SocketConfig({socket_connection}, socket_mem_config, local_mesh_id, recv_mesh_id);
            sockets.emplace(recv_mesh_id, MeshSocket(mesh_device, socket_config));
        }

        for (int i = 0; i < num_iterations; i++) {
            for (auto& [recv_mesh_id, socket] : sockets) {
                test_socket_send_recv(mesh_device, socket, data_size, socket_page_size);
            }
        }

    } else {
        SocketConfig socket_config =
            SocketConfig({socket_connection}, socket_mem_config, sender_mesh_id, local_mesh_id);
        auto socket = MeshSocket(mesh_device, socket_config);
        for (int i = 0; i < num_iterations; i++) {
            test_socket_send_recv(mesh_device, socket, data_size, socket_page_size);
        }
    }
    distributed_context->barrier();
}

void test_multi_mesh_single_conn_fwd(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t socket_fifo_size,
    uint32_t socket_page_size,
    uint32_t data_size,
    SystemConfig system_config) {
    using namespace tt::tt_metal::distributed::multihost;
    using namespace tt::tt_metal::distributed;
    using namespace tt_metal;

    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();

    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv_logical_coord = CoreCoord(0, 0);

    SocketConnection socket_connection(
        MeshCoreCoord(MeshCoordinate(0, 0), sender_logical_coord),
        MeshCoreCoord(MeshCoordinate(0, 0), recv_logical_coord));
    SocketMemoryConfig socket_mem_config(tt_metal::BufferType::L1, socket_fifo_size);

    auto local_mesh_id = get_local_mesh_id();
    constexpr tt::tt_fabric::MeshId recv_mesh_id = tt::tt_fabric::MeshId{1};
    constexpr uint32_t num_iterations = 50;

    if (local_mesh_id == recv_mesh_id) {
        std::unordered_map<tt::tt_fabric::MeshId, MeshSocket> sockets;
        std::vector<tt::tt_fabric::MeshId> sender_mesh_ids = get_neighbor_mesh_ids(system_config);

        for (const auto& sender_mesh_id : sender_mesh_ids) {
            SocketConfig socket_config =
                SocketConfig({socket_connection}, socket_mem_config, sender_mesh_id, local_mesh_id);
            sockets.emplace(sender_mesh_id, MeshSocket(mesh_device, socket_config));
        }
        for (int i = 0; i < num_iterations; i++) {
            for (auto& [sender_node, socket] : sockets) {
                test_socket_send_recv(mesh_device, socket, data_size, socket_page_size);
            }
        }
    } else {
        SocketConfig socket_config = SocketConfig({socket_connection}, socket_mem_config, local_mesh_id, recv_mesh_id);
        auto socket = MeshSocket(mesh_device, socket_config);
        for (int i = 0; i < num_iterations; i++) {
            test_socket_send_recv(mesh_device, socket, data_size, socket_page_size);
        }
    }
    distributed_context->barrier();
}

void test_multi_mesh_multi_conn_fwd(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t socket_fifo_size,
    uint32_t socket_page_size,
    uint32_t data_size,
    SystemConfig system_config) {
    using namespace tt::tt_metal::distributed::multihost;
    using namespace tt::tt_metal::distributed;
    using namespace tt_metal;

    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    std::vector<SocketConnection> socket_connections;
    auto sender_logical_core = CoreCoord(0, 0);
    auto recv_logical_core = CoreCoord(0, 0);

    for (const auto& coord : MeshCoordinateRange(mesh_device->shape())) {
        socket_connections.push_back(
            SocketConnection(MeshCoreCoord(coord, sender_logical_core), MeshCoreCoord(coord, recv_logical_core)));
    }

    SocketMemoryConfig socket_mem_config(BufferType::L1, socket_fifo_size);
    auto local_mesh_id = get_local_mesh_id();
    constexpr tt::tt_fabric::MeshId recv_mesh_id = tt::tt_fabric::MeshId{1};
    constexpr uint32_t num_iterations = 50;

    if (local_mesh_id == recv_mesh_id) {
        std::unordered_map<tt::tt_fabric::MeshId, MeshSocket> sockets;
        std::vector<tt::tt_fabric::MeshId> sender_mesh_ids = get_neighbor_mesh_ids(system_config);

        for (const auto& sender_mesh_id : sender_mesh_ids) {
            SocketConfig socket_config =
                SocketConfig({socket_connections}, socket_mem_config, sender_mesh_id, local_mesh_id);
            sockets.emplace(sender_mesh_id, MeshSocket(mesh_device, socket_config));
        }
        for (int i = 0; i < num_iterations; i++) {
            for (auto& [sender_node, socket] : sockets) {
                test_socket_send_recv(mesh_device, socket, data_size, socket_page_size);
            }
        }
    } else {
        SocketConfig socket_config = SocketConfig({socket_connections}, socket_mem_config, local_mesh_id, recv_mesh_id);
        auto socket = MeshSocket(mesh_device, socket_config);
        for (int i = 0; i < num_iterations; i++) {
            test_socket_send_recv(mesh_device, socket, data_size, socket_page_size);
        }
    }
    distributed_context->barrier();
}

void test_multi_mesh_multi_conn_bidirectional(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t socket_fifo_size,
    uint32_t socket_page_size,
    uint32_t data_size,
    SystemConfig system_config) {
    using namespace tt::tt_metal::distributed::multihost;
    using namespace tt::tt_metal::distributed;
    using namespace tt_metal;

    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    std::unordered_map<uint32_t, MeshSocket> forward_sockets;
    std::unordered_map<uint32_t, MeshSocket> backward_sockets;
    std::vector<SocketConnection> socket_connections;
    auto sender_logical_core = CoreCoord(0, 0);
    auto recv_logical_core = CoreCoord(0, 0);

    for (const auto& coord : MeshCoordinateRange(mesh_device->shape())) {
        socket_connections.push_back(
            SocketConnection(MeshCoreCoord(coord, sender_logical_core), MeshCoreCoord(coord, recv_logical_core)));
    }

    SocketMemoryConfig socket_mem_config(BufferType::L1, socket_fifo_size);
    auto local_mesh_id = get_local_mesh_id();
    constexpr tt::tt_fabric::MeshId aggregator_mesh_id = tt::tt_fabric::MeshId{1};
    constexpr uint32_t num_iterations = 50;

    if (local_mesh_id == aggregator_mesh_id) {
        std::unordered_map<tt::tt_fabric::MeshId, MeshSocket> forward_sockets;
        std::unordered_map<tt::tt_fabric::MeshId, MeshSocket> backward_sockets;
        std::vector<tt::tt_fabric::MeshId> compute_mesh_ids = get_neighbor_mesh_ids(system_config);

        for (const auto& compute_mesh_id : compute_mesh_ids) {
            SocketConfig forward_socket_config =
                SocketConfig({socket_connections}, socket_mem_config, compute_mesh_id, local_mesh_id);
            forward_sockets.emplace(compute_mesh_id, MeshSocket(mesh_device, forward_socket_config));

            SocketConfig backward_socket_config =
                SocketConfig({socket_connections}, socket_mem_config, local_mesh_id, compute_mesh_id);
            backward_sockets.emplace(compute_mesh_id, MeshSocket(mesh_device, backward_socket_config));
        }
        for (int i = 0; i < num_iterations; i++) {
            for (auto& [compute_node, forward_socket] : forward_sockets) {
                test_socket_send_recv(mesh_device, forward_socket, data_size, socket_page_size);
            }
            for (auto& [compute_node, backward_socket] : backward_sockets) {
                test_socket_send_recv(mesh_device, backward_socket, data_size, socket_page_size);
            }
        }
        for (int i = 0; i < num_iterations; i++) {
            for (auto& [compute_node, forward_socket] : forward_sockets) {
                test_socket_send_recv(mesh_device, forward_socket, data_size, socket_page_size);
            }
        }
        for (int i = 0; i < num_iterations; i++) {
            for (auto& [compute_node, backward_socket] : backward_sockets) {
                test_socket_send_recv(mesh_device, backward_socket, data_size, socket_page_size);
            }
        }
    } else {
        SocketConfig forward_socket_config =
            SocketConfig({socket_connections}, socket_mem_config, local_mesh_id, aggregator_mesh_id);
        auto forward_socket = MeshSocket(mesh_device, forward_socket_config);

        SocketConfig backward_socket_config =
            SocketConfig({socket_connections}, socket_mem_config, aggregator_mesh_id, local_mesh_id);
        auto backward_socket = MeshSocket(mesh_device, backward_socket_config);
        for (int i = 0; i < num_iterations; i++) {
            test_socket_send_recv(mesh_device, forward_socket, data_size, socket_page_size);
            test_socket_send_recv(mesh_device, backward_socket, data_size, socket_page_size);
        }

        for (int i = 0; i < num_iterations; i++) {
            test_socket_send_recv(mesh_device, forward_socket, data_size, socket_page_size);
        }
        for (int i = 0; i < num_iterations; i++) {
            test_socket_send_recv(mesh_device, backward_socket, data_size, socket_page_size);
        }
    }
    distributed_context->barrier();
}

}  // namespace tt::tt_fabric::fabric_router_tests::multihost::multihost_utils
