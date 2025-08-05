// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/multihost/fabric_tests/mesh_socket_test_runner.hpp"
#include "tests/tt_metal/multihost/fabric_tests/socket_send_recv_utils.hpp"

#include <stdexcept>
#include <algorithm>
#include <map>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/control_plane.hpp>
#include "tt_metal/fabric/fabric_context.hpp"
#include <tt-metalium/hal_types.hpp>

namespace tt::tt_fabric::mesh_socket_tests {

MeshSocketTestRunner::MeshSocketTestRunner(const MeshSocketTestConfiguration& config) :
    config_(config), expanded_tests_(), mesh_device_(nullptr), is_initialized_(false), control_plane_ptr_(nullptr) {
    log_info(tt::LogTest, "MeshSocketTestRunner created with {} tests", config_.tests.size());
}

MeshSocketTestRunner::~MeshSocketTestRunner() {
    if (is_initialized_) {
        cleanup();
    }
}

void MeshSocketTestRunner::initialize() {
    if (is_initialized_) {
        log_warning(tt::LogTest, "MeshSocketTestRunner already initialized");
        return;
    }

    log_info(tt::LogTest, "Initializing MeshSocketTestRunner...");

    // Initialize physical mesh if provided
    if (config_.physical_mesh_config.has_value()) {
        initialize_and_validate_custom_physical_config(config_.physical_mesh_config.value());
    }

    distributed_context_ = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    local_rank_ = distributed_context_->rank();
    log_info(tt::LogTest, "local_rank {}", *local_rank_);

    // Setup fabric configuration first
    setup_fabric_configuration();

    // Initialize control plane and get mesh shape
    control_plane_ptr_ = &tt::tt_metal::MetalContext::instance().get_control_plane();

    // Get mesh shape from control plane
    mesh_shape_ = control_plane_ptr_->get_physical_mesh_shape(
        control_plane_ptr_->get_user_physical_mesh_ids()[0], MeshScope::GLOBAL);

    // Expand test configurations now that mesh graph is set up
    expand_test_configurations();

    // Initialize MeshDevice
    initialize_mesh_device();

    is_initialized_ = true;
    log_info(tt::LogTest, "MeshSocketTestRunner initialization completed successfully");
}

void MeshSocketTestRunner::run_all_tests() {
    if (!is_initialized_) {
        throw std::runtime_error("MeshSocketTestRunner not initialized. Call initialize() first.");
    }

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

void MeshSocketTestRunner::cleanup() {
    log_info(tt::LogTest, "Cleaning up MeshSocketTestRunner...");

    if (mesh_device_) {
        mesh_device_->close();
        mesh_device_.reset();
    }

    is_initialized_ = false;
    log_info(tt::LogTest, "MeshSocketTestRunner cleanup completed");
}

std::shared_ptr<tt::tt_metal::distributed::MeshDevice> MeshSocketTestRunner::get_mesh_device() const {
    return mesh_device_;
}

const tt::tt_fabric::MeshGraph& MeshSocketTestRunner::get_mesh_graph() const {
    TT_FATAL(control_plane_ptr_, "Control plane not initialized");
    return control_plane_ptr_->get_mesh_graph();
}

const tt::tt_fabric::MeshId& MeshSocketTestRunner::get_local_mesh_id() const { return local_mesh_id_; }

void MeshSocketTestRunner::initialize_and_validate_custom_physical_config(
    const PhysicalMeshConfig& physical_mesh_config) {
    const auto mesh_id_str = std::string(std::getenv("TT_MESH_ID"));
    const auto host_rank_str = std::string(std::getenv("TT_HOST_RANK"));
    log_info(tt::LogTest, "host_rank_str: {}", host_rank_str);

    local_mesh_id_ = MeshId{std::stoi(mesh_id_str)};
    const auto local_rank = tt::tt_metal::distributed::multihost::Rank{std::stoi(host_rank_str)};

    const auto& eth_coord_mapping = physical_mesh_config.eth_coord_mapping;
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    // ethernet coordinate chip mapping, which should be migrated away from
    std::map<FabricNodeId, chip_id_t> chip_to_eth_coord_mapping;
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

void MeshSocketTestRunner::run_test(const ParsedTestConfig& test) {
    if (!should_participate_in_test(test)) {
        log_info(tt::LogTest, "Current rank not participating in test '{}'", test.name);
        return;
    }

    log_info(tt::LogTest, "Creating sockets for test '{}'...", test.name);
    auto sockets = create_sockets_for_test(test);

    // Log detailed socket information
    log_info(tt::LogTest, "Created {} socket(s) for test '{}'", sockets.size(), test.name);
    for (size_t i = 0; i < sockets.size(); ++i) {
        const auto& socket = sockets[i];
        const auto& config = socket.get_config();
        log_info(
            tt::LogTest,
            "  Socket {}: sender_rank={}, receiver_rank={}, connections={}",
            i + 1,
            *config.sender_rank,
            *config.receiver_rank,
            config.socket_connection_config.size());

        for (size_t j = 0; j < config.socket_connection_config.size(); ++j) {
            const auto& conn = config.socket_connection_config[j];
            log_info(
                tt::LogTest,
                "    Connection {}: sender_device=({},{}), receiver_device=({},{}), sender_core=({},{}), "
                "receiver_core=({},{})",
                j + 1,
                conn.sender_core.device_coord[0],
                conn.sender_core.device_coord[1],
                conn.receiver_core.device_coord[0],
                conn.receiver_core.device_coord[1],
                conn.sender_core.core_coord.x,
                conn.sender_core.core_coord.y,
                conn.receiver_core.core_coord.x,
                conn.receiver_core.core_coord.y);
        }

        log_info(
            tt::LogTest,
            "    Memory config: fifo_size={}, storage_type={}",
            config.socket_mem_config.fifo_size,
            config.socket_mem_config.socket_storage_type == tt::tt_metal::BufferType::L1 ? "L1" : "Other");
    }

    // Log test memory configuration
    log_info(
        tt::LogTest,
        "Test memory config: fifo_size={}, page_size={}, data_size={}, num_transactions={}",
        test.memory_config.fifo_size,
        test.memory_config.page_size,
        test.memory_config.data_size,
        test.memory_config.num_transactions);

    if (sockets.empty()) {
        log_warning(tt::LogTest, "No sockets created for test '{}' on current rank", test.name);
        return;
    }

    log_info(tt::LogTest, "Executing {} socket(s) for test '{}'", sockets.size(), test.name);

    uint32_t num_iterations = test.num_iterations.value_or(DEFAULT_NUM_ITERATIONS);

    for (uint32_t iteration = 0; iteration < num_iterations; ++iteration) {
        if (num_iterations > 1) {
            log_info(tt::LogTest, "--- Iteration {}/{} ---", iteration + 1, num_iterations);
        }

        for (size_t socket_idx = 0; socket_idx < sockets.size(); ++socket_idx) {
            log_test_execution(test, socket_idx, sockets.size());
            execute_socket_test(sockets[socket_idx], test);
        }
    }
    distributed_context_->barrier();
}

void MeshSocketTestRunner::initialize_mesh_device() {
    log_info(tt::LogTest, "Initializing MeshDevice...");

    // Create MeshDevice using the mesh shape obtained from control plane
    mesh_device_ =
        tt::tt_metal::distributed::MeshDevice::create(tt::tt_metal::distributed::MeshDeviceConfig(mesh_shape_));

    if (!mesh_device_) {
        throw std::runtime_error("Failed to create MeshDevice");
    }

    log_info(tt::LogTest, "MeshDevice created successfully with shape: {}", mesh_device_->shape());
}

void MeshSocketTestRunner::setup_fabric_configuration() {
    log_info(tt::LogTest, "Setting up fabric configuration...");

    // Convert our FabricConfig to tt::tt_fabric::FabricConfig
    tt::tt_fabric::FabricConfig fabric_config;

    switch (config_.fabric_config.topology) {
        case tt::tt_fabric::Topology::Mesh:
            switch (config_.fabric_config.routing_type) {
                case RoutingType::Dynamic: fabric_config = tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC; break;
                default: throw std::runtime_error("Unsupported fabric routing type, must be Dynamic");
            }
            break;
        default: throw std::runtime_error("Unsupported fabric topology, must be Mesh");
    }

    // Set the fabric configuration
    tt::tt_fabric::SetFabricConfig(fabric_config);
}

void MeshSocketTestRunner::expand_test_configurations() {
    log_info(tt::LogTest, "Expanding test configurations...");

    // Use the parser's expand_test_configs method, passing the test runner
    MeshSocketYamlParser parser;
    expanded_tests_ = parser.expand_test_configs(config_.tests, *this);

    log_info(
        tt::LogTest, "Expanded {} tests into {} test configurations", config_.tests.size(), expanded_tests_.size());
}

std::vector<tt::tt_metal::distributed::MeshSocket> MeshSocketTestRunner::create_sockets_for_test(
    const ParsedTestConfig& test) {
    std::vector<tt::tt_metal::distributed::MeshSocket> sockets;

    for (const auto& socket_config : test.sockets) {
        // Check if current rank is involved in this socket
        bool is_sender = (socket_config.sender_rank == local_rank_);
        bool is_receiver = (socket_config.receiver_rank == local_rank_);

        if (is_sender || is_receiver) {
            auto mesh_socket_config = convert_to_socket_config(socket_config, test.memory_config);
            sockets.emplace_back(mesh_device_, mesh_socket_config);

            log_info(tt::LogTest, "Created socket: rank {} as {}", local_rank_, is_sender ? "sender" : "receiver");
        }
    }

    return sockets;
}

tt::tt_metal::distributed::SocketConfig MeshSocketTestRunner::convert_to_socket_config(
    const TestSocketConfig& test_socket_config, const ParsedMemoryConfig& memory_config) {
    // Convert connections
    std::vector<tt::tt_metal::distributed::SocketConnection> connections;
    for (const auto& conn_config : test_socket_config.connections) {
        connections.push_back(convert_to_socket_connection(conn_config));
    }

    // Create memory configuration
    tt::tt_metal::distributed::SocketMemoryConfig socket_mem_config{
        .socket_storage_type = tt::tt_metal::BufferType::L1,
        .fifo_size = memory_config.fifo_size,
    };

    // Create distributed socket config
    tt::tt_metal::distributed::SocketConfig config{
        .socket_connection_config = connections,
        .socket_mem_config = socket_mem_config,
        .sender_rank = test_socket_config.sender_rank,
        .receiver_rank = test_socket_config.receiver_rank,
        .distributed_context = distributed_context_};

    return config;
}

tt::tt_metal::distributed::SocketConnection MeshSocketTestRunner::convert_to_socket_connection(
    const SocketConnectionConfig& connection_config) {
    return tt::tt_metal::distributed::SocketConnection{
        .sender_core = {connection_config.sender.mesh_coord, connection_config.sender.core_coord},
        .receiver_core = {connection_config.receiver.mesh_coord, connection_config.receiver.core_coord}};
}

void MeshSocketTestRunner::execute_socket_test(
    tt::tt_metal::distributed::MeshSocket& socket, const ParsedTestConfig& test) {
    // Use the existing test_socket_send_recv function from socket_send_recv_utils.cpp
    tt::tt_fabric::fabric_router_tests::multihost::multihost_utils::test_socket_send_recv(
        mesh_device_,
        socket,
        test.memory_config.data_size,
        test.memory_config.page_size,
        test.memory_config.num_transactions);
}

std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>
MeshSocketTestRunner::get_distributed_context() const {
    return distributed_context_;
}

bool MeshSocketTestRunner::should_participate_in_test(const ParsedTestConfig& test) const {
    // Check if current rank is involved in any socket of this test
    for (const auto& socket_config : test.sockets) {
        if (socket_config.sender_rank == local_rank_ || socket_config.receiver_rank == local_rank_) {
            return true;
        }
    }

    return false;
}

void MeshSocketTestRunner::log_test_execution(
    const ParsedTestConfig& test, size_t socket_index, size_t total_sockets) const {
    log_info(tt::LogTest, "Executing socket {}/{} for test '{}'", socket_index + 1, total_sockets, test.name);
    log_info(
        tt::LogTest,
        "  Data size: {} bytes, Page size: {} bytes, FIFO size: {} bytes",
        test.memory_config.data_size,
        test.memory_config.page_size,
        test.memory_config.fifo_size);
}

}  // namespace tt::tt_fabric::mesh_socket_tests
