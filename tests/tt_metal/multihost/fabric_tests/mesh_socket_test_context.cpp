// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/multihost/fabric_tests/mesh_socket_test_context.hpp"
#include "tests/tt_metal/multihost/fabric_tests/socket_send_recv_utils.hpp"

#include <algorithm>
#include <map>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "tt_metal/fabric/fabric_context.hpp"
#include <tt-metalium/hal_types.hpp>

namespace tt::tt_fabric::mesh_socket_tests {

MeshSocketTestContext::MeshSocketTestContext(const MeshSocketTestConfiguration& config) :
    config_(config), mesh_device_(nullptr) {
    log_info(tt::LogTest, "MeshSocketTestContext created with {} tests", config_.tests.size());
}

MeshSocketTestContext::~MeshSocketTestContext() { cleanup(); }

void MeshSocketTestContext::initialize() {
    log_info(tt::LogTest, "Initializing MeshSocketTestContext...");
    const auto mesh_id_str = std::string(std::getenv("TT_MESH_ID"));
    local_mesh_id_ = MeshId{std::stoi(mesh_id_str)};
    if (config_.physical_mesh_config.has_value()) {
        initialize_and_validate_custom_physical_config(config_.physical_mesh_config.value());
    }

    distributed_context_ = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();

    // Assert that we have more than one rank in the distributed world
    TT_FATAL(
        *distributed_context_->size() > 1,
        "Distributed world size must be greater than 1, but got {}",
        *distributed_context_->size());

    local_rank_ = distributed_context_->rank();
    log_info(tt::LogTest, "local_rank {}", *local_rank_);

    setup_fabric_configuration();
    control_plane_ptr_ = &tt::tt_metal::MetalContext::instance().get_control_plane();

    const auto mesh_shape = control_plane_ptr_->get_physical_mesh_shape(
        control_plane_ptr_->get_user_physical_mesh_ids()[0], MeshScope::GLOBAL);
    mesh_device_ =
        tt::tt_metal::distributed::MeshDevice::create(tt::tt_metal::distributed::MeshDeviceConfig(mesh_shape));
    TT_FATAL(mesh_device_, "Failed to create MeshDevice");
    log_info(tt::LogTest, "MeshDevice created successfully with shape: {}", mesh_device_->shape());

    rank_to_mesh_mapping_ = create_rank_to_mesh_mapping();
    share_seed();
    expand_test_configurations();

    log_info(tt::LogTest, "MeshSocketTestContext initialization completed successfully");
}

void MeshSocketTestContext::run_all_tests() {
    log_info(tt::LogTest, "Running all {} expanded tests...", expanded_tests_.size());

    for (size_t i = 0; i < expanded_tests_.size(); ++i) {
        const auto& test = expanded_tests_[i];
        log_info(tt::LogTest, "=== Running Test {}/{}: '{}' ===", i + 1, expanded_tests_.size(), test.name);

        run_test(test);

        Finish(mesh_device_->mesh_command_queue(0));
        log_info(tt::LogTest, "✓ Test '{}' completed successfully", test.name);
    }

    log_info(tt::LogTest, "All {} tests completed successfully!", expanded_tests_.size());
}

void MeshSocketTestContext::cleanup() {
    log_info(tt::LogTest, "Cleaning up MeshSocketTestContext...");

    if (mesh_device_) {
        mesh_device_->close();
        mesh_device_.reset();
    }
    log_info(tt::LogTest, "MeshSocketTestContext cleanup completed");
}

const tt::tt_fabric::MeshGraph& MeshSocketTestContext::get_mesh_graph() const {
    TT_FATAL(control_plane_ptr_, "Control plane not initialized");
    return control_plane_ptr_->get_mesh_graph();
}

const std::unordered_map<Rank, tt::tt_fabric::MeshId>& MeshSocketTestContext::get_rank_to_mesh_mapping() const {
    return rank_to_mesh_mapping_;
}

const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>&
MeshSocketTestContext::get_distributed_context() const {
    return distributed_context_;
}

const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& MeshSocketTestContext::get_mesh_device() const {
    TT_FATAL(mesh_device_, "Mesh device not initialized");
    return mesh_device_;
}

void MeshSocketTestContext::initialize_and_validate_custom_physical_config(
    const PhysicalMeshConfig& physical_mesh_config) {
    const auto& eth_coord_mapping = physical_mesh_config.eth_coord_mapping;
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    // ethernet coordinate chip mapping, which should be migrated away from
    std::map<FabricNodeId, ChipId> chip_to_eth_coord_mapping;
    for (std::uint32_t mesh_id = 0; mesh_id < eth_coord_mapping.size(); mesh_id++) {
        if (mesh_id == *local_mesh_id_) {
            for (std::uint32_t chip_id = 0; chip_id < eth_coord_mapping[mesh_id].size(); chip_id++) {
                const auto& eth_coord = eth_coord_mapping[mesh_id][chip_id];
                chip_to_eth_coord_mapping.insert(
                    {FabricNodeId(MeshId{mesh_id}, chip_id), cluster.get_physical_chip_id_from_eth_coord(eth_coord)});
            }
        }
    }
    tt::tt_metal::MetalContext::instance().set_custom_fabric_topology(
        physical_mesh_config.mesh_descriptor_path, chip_to_eth_coord_mapping);
}

void MeshSocketTestContext::run_test(const ParsedTestConfig& test) {
    if (!should_participate_in_test(test)) {
        log_info(tt::LogTest, "Current rank not participating in test '{}'", test.name);
        return;
    }

    log_info(tt::LogTest, "Creating sockets for test '{}'...", test.name);
    auto sockets = create_sockets_for_test(test);

    log_info(tt::LogTest, "Executing {} socket(s) for test '{}'", sockets.size(), test.name);

    uint32_t num_iterations = test.num_iterations.value_or(DEFAULT_NUM_ITERATIONS);

    for (uint32_t iteration = 0; iteration < num_iterations; ++iteration) {
        log_info(tt::LogTest, "--- Iteration {}/{} ---", iteration + 1, num_iterations);

        for (size_t socket_idx = 0; socket_idx < sockets.size(); ++socket_idx) {
            log_info(tt::LogTest, "Executing socket {}/{}", socket_idx + 1, sockets.size());
            execute_socket_test(sockets[socket_idx], test);
        }
    }
    distributed_context_->barrier();
}

// NOLINTNEXTLINE(readability-make-member-function-const)
void MeshSocketTestContext::setup_fabric_configuration() {
    log_info(tt::LogTest, "Setting up fabric configuration...");

    tt::tt_fabric::FabricConfig fabric_config;
    // TODO: Add support for other Fabric Configs as well
    switch (config_.fabric_config.topology) {
        case tt::tt_fabric::Topology::Mesh: {
            fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D;
        } break;
        default: TT_THROW("Unsupported fabric topology, must be Mesh");
    }

    tt::tt_fabric::SetFabricConfig(fabric_config);
}

void MeshSocketTestContext::expand_test_configurations() {
    log_info(tt::LogTest, "Expanding test configurations...");

    expanded_tests_ = MeshSocketYamlParser::expand_test_configs(config_.tests, *this);

    log_info(
        tt::LogTest, "Expanded {} tests into {} test configurations", config_.tests.size(), expanded_tests_.size());
}

std::vector<tt::tt_metal::distributed::MeshSocket> MeshSocketTestContext::create_sockets_for_test(
    const ParsedTestConfig& test) {
    std::vector<tt::tt_metal::distributed::MeshSocket> sockets;

    for (const auto& socket_config : test.sockets) {
        bool is_sender = (socket_config.sender_rank == local_rank_);
        bool is_receiver = (socket_config.receiver_rank == local_rank_);

        if (is_sender || is_receiver) {
            auto mesh_socket_config = convert_to_socket_config(socket_config, test.memory_config);
            sockets.emplace_back(mesh_device_, mesh_socket_config);
        }
    }

    return sockets;
}

tt::tt_metal::distributed::SocketConfig MeshSocketTestContext::convert_to_socket_config(
    const TestSocketConfig& test_socket_config, const ParsedMemoryConfig& memory_config) {
    std::vector<tt::tt_metal::distributed::SocketConnection> connections;
    connections.reserve(test_socket_config.connections.size());
    for (const auto& conn_config : test_socket_config.connections) {
        connections.push_back(convert_to_socket_connection(conn_config));
    }

    tt::tt_metal::distributed::SocketMemoryConfig socket_mem_config(
        tt::tt_metal::BufferType::L1, memory_config.fifo_size);

    tt::tt_metal::distributed::SocketConfig config(
        connections,
        socket_mem_config,
        test_socket_config.sender_rank,
        test_socket_config.receiver_rank,
        distributed_context_);

    return config;
}

tt::tt_metal::distributed::SocketConnection MeshSocketTestContext::convert_to_socket_connection(
    const SocketConnectionConfig& connection_config) {
    return tt::tt_metal::distributed::SocketConnection(
        tt::tt_metal::distributed::MeshCoreCoord(
            connection_config.sender.mesh_coord, connection_config.sender.core_coord),
        tt::tt_metal::distributed::MeshCoreCoord(
            connection_config.receiver.mesh_coord, connection_config.receiver.core_coord));
}

void MeshSocketTestContext::execute_socket_test(
    tt::tt_metal::distributed::MeshSocket& socket, const ParsedTestConfig& test) {
    // Use the existing test_socket_send_recv function from socket_send_recv_utils.cpp
    TT_FATAL(
        tt::tt_fabric::fabric_router_tests::multihost::multihost_utils::test_socket_send_recv(
            mesh_device_,
            socket,
            test.memory_config.data_size,
            test.memory_config.page_size,
            test.memory_config.num_transactions,
            gen_),
        "Socket test {} failed",
        test.name);
}

bool MeshSocketTestContext::should_participate_in_test(const ParsedTestConfig& test) const {
    for (const auto& socket_config : test.sockets) {
        if (socket_config.sender_rank == local_rank_ || socket_config.receiver_rank == local_rank_) {
            return true;
        }
    }
    return false;
}

/*
    We assume rank to mesh is 1-to-1, each rank sends its mesh_id and we receive all mesh_ids
    Sockets APIs will need to change to use mesh_id instead of rank to supprot Big Mesh x Multi Mesh case
    Need this to generate high level patterns such as all to all, since we need to know the mesh_ id to
    know the number of devices per host.
*/
std::unordered_map<Rank, tt::tt_fabric::MeshId> MeshSocketTestContext::create_rank_to_mesh_mapping() {
    auto world_size = *distributed_context_->size();

    std::vector<std::byte> recv_buffer(sizeof(uint32_t) * world_size);
    distributed_context_->all_gather(
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&local_mesh_id_), sizeof(local_mesh_id_)),
        tt::stl::Span<std::byte>(recv_buffer));

    std::unordered_map<Rank, tt::tt_fabric::MeshId> rank_to_mesh_id;
    for (uint32_t rank = 0; rank < world_size; ++rank) {
        uint32_t mesh_id_val;
        std::memcpy(&mesh_id_val, recv_buffer.data() + (rank * sizeof(uint32_t)), sizeof(uint32_t));
        log_info(tt::LogTest, "Rank {} is in mesh {}", rank, mesh_id_val);
        TT_FATAL(
            !std::any_of(
                rank_to_mesh_id.begin(),
                rank_to_mesh_id.end(),
                [&mesh_id_val](const auto& pair) { return *(pair.second) == mesh_id_val; }),
            "Mesh id {} is already in use",
            mesh_id_val);
        rank_to_mesh_id[Rank{rank}] = tt::tt_fabric::MeshId{mesh_id_val};
    }

    return rank_to_mesh_id;
}

void MeshSocketTestContext::share_seed() {
    uint32_t seed;

    if (*local_rank_ == 0) {
        // Rank 0 generates and sends the seed
        seed = std::chrono::steady_clock::now().time_since_epoch().count();
        log_info(tt::LogTest, "Rank 0 generated seed: {}", seed);

        // Send seed to all other ranks
        for (uint32_t rank = 1; rank < *distributed_context_->size(); ++rank) {
            distributed_context_->send(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&seed), sizeof(seed)),
                tt::tt_metal::distributed::multihost::Rank{rank},
                tt::tt_metal::distributed::multihost::Tag{0});
        }
    } else {
        // All other ranks receive the seed from rank 0
        log_info(tt::LogTest, "Rank {} receiving seed from rank 0", *local_rank_);
        distributed_context_->recv(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&seed), sizeof(seed)),
            tt::tt_metal::distributed::multihost::Rank{0},
            tt::tt_metal::distributed::multihost::Tag{0});
        log_info(tt::LogTest, "Rank {} received seed: {}", *local_rank_, seed);
    }

    // Initialize the random number generator with the shared seed
    gen_.seed(seed);
    log_info(tt::LogTest, "Random number generator initialized with seed: {}", seed);
}

}  // namespace tt::tt_fabric::mesh_socket_tests
