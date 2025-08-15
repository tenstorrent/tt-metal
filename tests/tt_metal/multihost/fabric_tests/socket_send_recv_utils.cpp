// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <stdint.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/fabric.hpp>

#include <algorithm>

#include "tests/tt_metal/multihost/fabric_tests/socket_send_recv_utils.hpp"

namespace tt::tt_fabric {
namespace fabric_router_tests::multihost {

namespace multihost_utils {

std::string get_system_config_name(SystemConfig system_config) {
    switch (system_config) {
        case SystemConfig::SPLIT_T3K: return "SplitT3K";
        case SystemConfig::DUAL_T3K: return "DualT3K";
        case SystemConfig::NANO_EXABOX: return "NanoExabox";
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

    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    auto sender_rank = socket.get_config().sender_rank;
    auto recv_rank = socket.get_config().receiver_rank;
    std::set<CoreRange> sender_core_range;
    std::set<CoreRange> recv_core_range;
    for (const auto& connection : socket.get_config().socket_connection_config) {
        sender_core_range.insert(CoreRange(connection.sender_core.core_coord, connection.sender_core.core_coord));
        recv_core_range.insert(CoreRange(connection.receiver_core.core_coord, connection.receiver_core.core_coord));
    }
    auto sender_core_range_set = CoreRangeSet(sender_core_range);
    auto recv_core_range_set = CoreRangeSet(recv_core_range);

    std::vector<uint32_t> src_vec_per_core(data_size / sizeof(uint32_t));

    if (!gen.has_value()) {
        // Exchange seed between sender and receiver and create local generator
        uint32_t seed;
        if (distributed_context->rank() == sender_rank) {
            seed = std::chrono::steady_clock::now().time_since_epoch().count();
            log_info(tt::LogTest, "Sending seed {} to rank {}", seed, *recv_rank);
            distributed_context->send(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&seed), sizeof(seed)),
                recv_rank,                                    // send to receiver host
                tt::tt_metal::distributed::multihost::Tag{0}  // exchange seed over tag 0
            );
        } else if (distributed_context->rank() == recv_rank) {
            log_info(tt::LogTest, "Receiving seed from rank {}", *sender_rank);
            distributed_context->recv(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&seed), sizeof(seed)),
                sender_rank,                                  // recv from sender host
                tt::tt_metal::distributed::multihost::Tag{0}  // exchange seed over tag 0
            );
        }
        log_info(tt::LogTest, "Using seed: {}", seed);
        gen = std::mt19937(seed);
    }
    std::uniform_int_distribution<uint32_t> dis(0, UINT32_MAX);
    std::generate(src_vec_per_core.begin(), src_vec_per_core.end(), [&]() { return dis(gen.value()); });
    std::vector<uint32_t> src_vec;
    src_vec.reserve(data_size * sender_core_range_set.num_cores() / sizeof(uint32_t));

    // duplicate data for all cores; this is non-ideal but there is no elegant way to not do this with current APIs
    for (int i = 0; i < sender_core_range_set.num_cores(); i++) {
        src_vec.insert(src_vec.end(), src_vec_per_core.begin(), src_vec_per_core.end());
    }
    const auto reserved_packet_header_CB_index = tt::CB::c_in0;

    for (int i = 0; i < num_txns; i++) {
        if (distributed_context->rank() == sender_rank) {
            auto sender_data_shard_params = ShardSpecBuffer(
                sender_core_range, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {sender_core_range_set.num_cores(), 1});

            const DeviceLocalBufferConfig sender_device_local_config{
                .page_size = data_size,
                .buffer_type = BufferType::L1,
                .sharding_args = BufferShardingArgs(sender_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
                .bottom_up = false};
            const ReplicatedBufferConfig buffer_config{.size = sender_core_range_set.num_cores() * data_size};

            auto sender_data_buffer = MeshBuffer::create(buffer_config, sender_device_local_config, mesh_device_.get());
            auto sender_mesh_workload = CreateMeshWorkload();

            for (const auto& connection : socket.get_config().socket_connection_config) {
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

                auto sender_packet_header_CB_handle =
                    CreateCircularBuffer(sender_program, sender_core, sender_cb_reserved_packet_header_config);

                std::vector<uint32_t> sender_rtas;
                tt_fabric::append_fabric_connection_rt_args(
                    sender_fabric_node_id, recv_fabric_node_id, 0, sender_program, {sender_core}, sender_rtas);

                tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_core, sender_rtas);
                AddProgramToMeshWorkload(
                    sender_mesh_workload,
                    std::move(sender_program),
                    MeshCoordinateRange(connection.sender_core.device_coord));
            }
            // Run workload performing Data Movement over the socket
            EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), sender_mesh_workload, false);
            Finish(mesh_device_->mesh_command_queue());
        } else if (distributed_context->rank() == recv_rank) {
            auto recv_data_shard_params = ShardSpecBuffer(
                recv_core_range, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {recv_core_range_set.num_cores(), 1});

            const DeviceLocalBufferConfig recv_device_local_config{
                .page_size = data_size,
                .buffer_type = BufferType::L1,
                .sharding_args = BufferShardingArgs(recv_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
                .bottom_up = false};

            const ReplicatedBufferConfig buffer_config{.size = recv_core_range_set.num_cores() * data_size};
            auto recv_data_buffer = MeshBuffer::create(buffer_config, recv_device_local_config, mesh_device_.get());

            auto recv_mesh_workload = CreateMeshWorkload();
            for (const auto& connection : socket.get_config().socket_connection_config) {
                auto recv_core = connection.receiver_core.core_coord;
                auto sender_fabric_node_id =
                    socket.get_fabric_node_id(SocketEndpoint::SENDER, connection.sender_core.device_coord);
                auto recv_fabric_node_id = mesh_device_->get_fabric_node_id(connection.receiver_core.device_coord);

                auto recv_program = CreateProgram();

                auto recv_virtual_coord = recv_data_buffer->device()->worker_core_from_logical_core(recv_core);
                auto output_virtual_coord = recv_data_buffer->device()->worker_core_from_logical_core(recv_core);

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
                AddProgramToMeshWorkload(
                    recv_mesh_workload,
                    std::move(recv_program),
                    MeshCoordinateRange(connection.receiver_core.device_coord));
            }
            // Run receiver workload using the created socket
            EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), recv_mesh_workload, false);
            auto& core_to_core_id = recv_data_buffer->get_backing_buffer()->get_buffer_page_mapping()->core_to_core_id;
            for (const auto& connection : socket.get_config().socket_connection_config) {
                std::vector<uint32_t> recv_data_readback;
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
                EXPECT_TRUE(is_data_match);
            }
        }
        // Increment the source vector for the next iteration
        // This is to ensure that the data is different for each transaction
        for (int i = 0; i < src_vec.size(); i++) {
            src_vec[i]++;
        }
        for (int i = 0; i < src_vec_per_core.size(); i++) {
            src_vec_per_core[i]++;
        }
    }
    return is_data_match;
}

std::vector<uint32_t> get_neighbor_host_ranks(SystemConfig system_config) {
    std::vector<uint32_t> recv_ranks;

    if (system_config == SystemConfig::NANO_EXABOX) {
        // Nano-Exabox has 5 hosts. Sender ranks assignment is customized for a particular Rank File.
        recv_ranks = {0, 2, 3, 4};
    } else if (system_config == SystemConfig::SPLIT_T3K || system_config == SystemConfig::DUAL_T3K) {
        // Only a single recv node is needed for the dual host configurations.
        recv_ranks = {0};
    } else {
        TT_THROW("Unsupported system configuration for multi-mesh single connection test.");
    }
    return recv_ranks;
}

void test_multi_mesh_single_conn_bwd(
    std::shared_ptr<tt_metal::distributed::MeshDevice> mesh_device,
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

    SocketConnection socket_connection = {
        .sender_core = {MeshCoordinate(0, 0), sender_logical_coord},
        .receiver_core = {MeshCoordinate(0, 0), recv_logical_coord}};

    SocketMemoryConfig socket_mem_config = {
        .socket_storage_type = tt_metal::BufferType::L1,
        .fifo_size = socket_fifo_size,
    };

    constexpr uint32_t sender_rank = 1;
    constexpr uint32_t num_iterations = 50;

    if (*distributed_context->rank() == sender_rank) {
        std::unordered_map<uint32_t, MeshSocket> sockets;
        std::vector<uint32_t> recv_node_ranks = get_neighbor_host_ranks(system_config);

        for (const auto& recv_rank : recv_node_ranks) {
            SocketConfig socket_config = {
                .socket_connection_config = {socket_connection},
                .socket_mem_config = socket_mem_config,
                .sender_rank = distributed_context->rank(),
                .receiver_rank = tt::tt_metal::distributed::multihost::Rank{recv_rank}};
            sockets.emplace(recv_rank, MeshSocket(mesh_device, socket_config));
        }

        for (int i = 0; i < num_iterations; i++) {
            for (auto& [recv_node, socket] : sockets) {
                test_socket_send_recv(mesh_device, socket, data_size, socket_page_size);
            }
        }

    } else {
        SocketConfig socket_config = {
            .socket_connection_config = {socket_connection},
            .socket_mem_config = socket_mem_config,
            .sender_rank = tt::tt_metal::distributed::multihost::Rank{sender_rank},
            .receiver_rank = distributed_context->rank()};
        auto socket = MeshSocket(mesh_device, socket_config);
        for (int i = 0; i < num_iterations; i++) {
            test_socket_send_recv(mesh_device, socket, data_size, socket_page_size);
        }
    }
    distributed_context->barrier();
}

void test_multi_mesh_single_conn_fwd(
    std::shared_ptr<tt_metal::distributed::MeshDevice> mesh_device,
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

    SocketConnection socket_connection = {
        .sender_core = {MeshCoordinate(0, 0), sender_logical_coord},
        .receiver_core = {MeshCoordinate(0, 0), recv_logical_coord}};
    SocketMemoryConfig socket_mem_config = {
        .socket_storage_type = tt_metal::BufferType::L1,
        .fifo_size = socket_fifo_size,
    };

    constexpr uint32_t recv_rank = 1;
    constexpr uint32_t num_iterations = 50;

    if (*distributed_context->rank() == recv_rank) {
        std::unordered_map<uint32_t, MeshSocket> sockets;
        std::vector<uint32_t> sender_node_ranks = get_neighbor_host_ranks(system_config);

        for (const auto& sender_rank : sender_node_ranks) {
            SocketConfig socket_config = {
                .socket_connection_config = {socket_connection},
                .socket_mem_config = socket_mem_config,
                .sender_rank = tt::tt_metal::distributed::multihost::Rank{sender_rank},
                .receiver_rank = distributed_context->rank()};
            sockets.emplace(sender_rank, MeshSocket(mesh_device, socket_config));
        }
        for (int i = 0; i < num_iterations; i++) {
            for (auto& [sender_node, socket] : sockets) {
                test_socket_send_recv(mesh_device, socket, data_size, socket_page_size);
            }
        }
    } else {
        SocketConfig socket_config = {
            .socket_connection_config = {socket_connection},
            .socket_mem_config = socket_mem_config,
            .sender_rank = distributed_context->rank(),
            .receiver_rank = tt::tt_metal::distributed::multihost::Rank{recv_rank}};
        auto socket = MeshSocket(mesh_device, socket_config);
        for (int i = 0; i < num_iterations; i++) {
            test_socket_send_recv(mesh_device, socket, data_size, socket_page_size);
        }
    }
    distributed_context->barrier();
}

void test_multi_mesh_multi_conn_fwd(
    std::shared_ptr<tt_metal::distributed::MeshDevice> mesh_device,
    uint32_t socket_fifo_size,
    uint32_t socket_page_size,
    uint32_t data_size,
    SystemConfig system_config) {
    using namespace tt::tt_metal::distributed::multihost;
    using namespace tt::tt_metal::distributed;
    using namespace tt_metal;

    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    std::unordered_map<uint32_t, MeshSocket> sockets;
    std::vector<SocketConnection> socket_connections;
    auto sender_logical_core = CoreCoord(0, 0);
    auto recv_logical_core = CoreCoord(0, 0);

    for (const auto& coord : MeshCoordinateRange(mesh_device->shape())) {
        socket_connections.push_back(
            {.sender_core = {coord, sender_logical_core}, .receiver_core = {coord, recv_logical_core}});
    }

    SocketMemoryConfig socket_mem_config = {
        .socket_storage_type = BufferType::L1,
        .fifo_size = socket_fifo_size,
    };
    constexpr uint32_t recv_rank = 1;
    constexpr uint32_t num_iterations = 50;

    if (*distributed_context->rank() == recv_rank) {
        std::unordered_map<uint32_t, MeshSocket> sockets;
        std::vector<uint32_t> sender_node_ranks = get_neighbor_host_ranks(system_config);

        for (const auto& sender_rank : sender_node_ranks) {
            SocketConfig socket_config = {
                .socket_connection_config = {socket_connections},
                .socket_mem_config = socket_mem_config,
                .sender_rank = tt::tt_metal::distributed::multihost::Rank{sender_rank},
                .receiver_rank = distributed_context->rank()};
            sockets.emplace(sender_rank, MeshSocket(mesh_device, socket_config));
        }
        for (int i = 0; i < num_iterations; i++) {
            for (auto& [sender_node, socket] : sockets) {
                test_socket_send_recv(mesh_device, socket, data_size, socket_page_size);
            }
        }
    } else {
        SocketConfig socket_config = {
            .socket_connection_config = {socket_connections},
            .socket_mem_config = socket_mem_config,
            .sender_rank = distributed_context->rank(),
            .receiver_rank = tt::tt_metal::distributed::multihost::Rank{recv_rank},
        };
        auto socket = MeshSocket(mesh_device, socket_config);
        for (int i = 0; i < num_iterations; i++) {
            test_socket_send_recv(mesh_device, socket, data_size, socket_page_size);
        }
    }
    distributed_context->barrier();
}

void test_multi_mesh_multi_conn_bidirectional(
    std::shared_ptr<tt_metal::distributed::MeshDevice> mesh_device,
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
            {.sender_core = {coord, sender_logical_core}, .receiver_core = {coord, recv_logical_core}});
    }

    SocketMemoryConfig socket_mem_config = {
        .socket_storage_type = BufferType::L1,
        .fifo_size = socket_fifo_size,
    };
    constexpr uint32_t aggregator_rank = 1;
    constexpr uint32_t num_iterations = 50;

    if (*distributed_context->rank() == aggregator_rank) {
        std::unordered_map<uint32_t, MeshSocket> forward_sockets;
        std::unordered_map<uint32_t, MeshSocket> backward_sockets;
        std::vector<uint32_t> compute_node_ranks = get_neighbor_host_ranks(system_config);

        for (const auto& compute_rank : compute_node_ranks) {
            SocketConfig forward_socket_config = {
                .socket_connection_config = {socket_connections},
                .socket_mem_config = socket_mem_config,
                .sender_rank = tt::tt_metal::distributed::multihost::Rank{compute_rank},
                .receiver_rank = distributed_context->rank()};
            forward_sockets.emplace(compute_rank, MeshSocket(mesh_device, forward_socket_config));

            SocketConfig backward_socket_config = {
                .socket_connection_config = {socket_connections},
                .socket_mem_config = socket_mem_config,
                .sender_rank = distributed_context->rank(),
                .receiver_rank = tt::tt_metal::distributed::multihost::Rank{compute_rank}};
            backward_sockets.emplace(compute_rank, MeshSocket(mesh_device, backward_socket_config));
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
        SocketConfig forward_socket_config = {
            .socket_connection_config = {socket_connections},
            .socket_mem_config = socket_mem_config,
            .sender_rank = distributed_context->rank(),
            .receiver_rank = tt::tt_metal::distributed::multihost::Rank{aggregator_rank}};
        auto forward_socket = MeshSocket(mesh_device, forward_socket_config);

        SocketConfig backward_socket_config = {
            .socket_connection_config = {socket_connections},
            .socket_mem_config = socket_mem_config,
            .sender_rank = tt::tt_metal::distributed::multihost::Rank{aggregator_rank},
            .receiver_rank = distributed_context->rank()};
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

}  // namespace multihost_utils

}  // namespace fabric_router_tests::multihost
}  // namespace tt::tt_fabric
