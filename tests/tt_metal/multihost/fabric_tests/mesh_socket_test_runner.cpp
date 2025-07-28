// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/multihost/fabric_tests/mesh_socket_test_runner.hpp"
#include "tests/tt_metal/multihost/fabric_tests/socket_send_recv_utils.hpp"

#include <stdexcept>
#include <algorithm>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/system_mesh.hpp>

namespace tt::tt_fabric::mesh_socket_tests {

MeshSocketTestRunner::MeshSocketTestRunner(const MeshSocketTestConfiguration& config) :
    config_(config), mesh_device_(nullptr), is_initialized_(false) {
    distributed_context_ = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    local_rank_ = distributed_context_->rank();
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

    try {
        log_info(tt::LogTest, "Initializing MeshSocketTestRunner...");

        // Setup fabric configuration first
        setup_fabric_configuration();

        // Initialize MeshDevice
        initialize_mesh_device();

        is_initialized_ = true;
        log_info(tt::LogTest, "MeshSocketTestRunner initialization completed successfully");

    } catch (const std::exception& e) {
        log_error(tt::LogTest, "Failed to initialize MeshSocketTestRunner: {}", e.what());
        cleanup();
        throw std::runtime_error("MeshSocketTestRunner initialization failed: " + std::string(e.what()));
    }
}

void MeshSocketTestRunner::run_all_tests() {
    if (!is_initialized_) {
        throw std::runtime_error("MeshSocketTestRunner not initialized. Call initialize() first.");
    }

    log_info(tt::LogTest, "Running all {} tests...", config_.tests.size());

    for (size_t i = 0; i < config_.tests.size(); ++i) {
        const auto& test = config_.tests[i];
        log_info(tt::LogTest, "=== Running Test {}/{}: '{}' ===", i + 1, config_.tests.size(), test.name);

        try {
            run_test(test);
            log_info(tt::LogTest, "✓ Test '{}' completed successfully", test.name);
        } catch (const std::exception& e) {
            log_error(tt::LogTest, "✗ Test '{}' failed: {}", test.name, e.what());
            throw;
        }
    }

    log_info(tt::LogTest, "All tests completed successfully!");
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

void MeshSocketTestRunner::run_test(const ParsedTestConfig& test) {
    if (!should_participate_in_test(test)) {
        log_info(tt::LogTest, "Current rank not participating in test '{}'", test.name);
        return;
    }

    log_info(tt::LogTest, "Creating sockets for test '{}'...", test.name);
    auto sockets = create_sockets_for_test(test);

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
}

void MeshSocketTestRunner::initialize_mesh_device() {
    log_info(tt::LogTest, "Initializing MeshDevice...");

    // Create MeshDevice - the distributed context and device pool should already be set up
    // The MeshDevice will use the default mesh shape from the system
    mesh_device_ = tt::tt_metal::distributed::MeshDevice::create(tt::tt_metal::distributed::MeshDeviceConfig(
        tt::tt_metal::distributed::MeshShape{2, 2}  // Default 2x2 for now, can be made configurable
        ));

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

std::vector<tt::tt_metal::distributed::MeshSocket> MeshSocketTestRunner::create_sockets_for_test(
    const ParsedTestConfig& test) {
    std::vector<tt::tt_metal::distributed::MeshSocket> sockets;

    for (const auto& socket_config : test.sockets) {
        // Check if current rank is involved in this socket
        bool is_sender = (socket_config.sender_rank == local_rank_);
        bool is_receiver = (socket_config.receiver_rank == local_rank_);

        if (is_sender || is_receiver) {
            try {
                auto mesh_socket_config = convert_to_socket_config(socket_config, test.memory_config);
                sockets.emplace_back(mesh_device_, mesh_socket_config);

                log_info(tt::LogTest, "Created socket: rank {} as {}", local_rank_, is_sender ? "sender" : "receiver");
            } catch (const std::exception& e) {
                log_error(tt::LogTest, "Failed to create socket: {}", e.what());
                throw;
            }
        }
    }

    return sockets;
}

tt::tt_metal::distributed::SocketConfig MeshSocketTestRunner::convert_to_socket_config(
    const TestSocketConfig& test_socket_config, const MemoryConfig& memory_config) {
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
        mesh_device_, socket, test.memory_config.data_size, test.memory_config.page_size, DEFAULT_NUM_TRANSACTIONS);
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
