// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/fabric.hpp>

#include "tests/tt_metal/multihost/fabric_tests/mesh_socket_yaml_parser.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "impl/context/metal_context.hpp"

using MeshId = tt::tt_fabric::MeshId;
using MeshShape = tt::tt_fabric::MeshShape;
using ControlPlane = tt::tt_fabric::ControlPlane;

namespace tt::tt_fabric::mesh_socket_tests {

/**
 * @brief Main test runner class that manages MeshDevice and MeshSocket lifetimes
 * and orchestrates socket test execution using the existing socket_send_recv_utils.
 */
class MeshSocketTestRunner {
public:
    /**
     * @brief Construct a new MeshSocketTestRunner
     *
     * @param config The parsed YAML configuration containing test definitions
     */
    explicit MeshSocketTestRunner(const MeshSocketTestConfiguration& config);

    /**
     * @brief Destructor - ensures proper cleanup of all resources
     */
    ~MeshSocketTestRunner();

    // Delete copy constructor and assignment operator to prevent accidental copying
    MeshSocketTestRunner(const MeshSocketTestRunner&) = delete;
    MeshSocketTestRunner& operator=(const MeshSocketTestRunner&) = delete;

    /**
     * @brief Initialize the test environment and MeshDevice
     *
     * @throws std::runtime_error if initialization fails
     */
    void initialize();

    /**
     * @brief Run all tests defined in the configuration
     *
     * @throws std::runtime_error if any test fails
     */
    void run_all_tests();

    /**
     * @brief Run a specific test by name
     *
     * @param test_name Name of the test to run
     * @throws std::runtime_error if test not found or execution fails
     */
    void run_test_by_name(const std::string& test_name);

    /**
     * @brief Clean up all resources and close MeshDevice
     */
    void cleanup();

    /**
     * @brief Get the current MeshDevice (for advanced usage)
     *
     * @return std::shared_ptr<tt::tt_metal::distributed::MeshDevice>
     */
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> get_mesh_device() const;

    /**
     * @brief Get the mesh graph
     *
     * @return const tt::tt_fabric::MeshGraph&
     */
    const tt::tt_fabric::MeshGraph& get_mesh_graph() const;

    /**
     * @brief Get the local mesh ID
     *
     * @return const tt::tt_fabric::MeshId&
     */
    const tt::tt_fabric::MeshId& get_local_mesh_id() const;

    /**
     * @brief Get the rank to mesh mapping
     *
     * @return const std::unordered_map<Rank, tt::tt_fabric::MeshId>&
     */
    const std::unordered_map<Rank, tt::tt_fabric::MeshId>& get_rank_to_mesh_mapping() const;

private:
    void initialize_and_validate_custom_physical_config(const PhysicalMeshConfig& physical_mesh_config);

    /**
     * @brief Run a specific test configuration
     *
     * @param test The test configuration to execute
     */
    void run_test(const ParsedTestConfig& test);

    /**
     * @brief Initialize the MeshDevice based on fabric configuration
     */
    void initialize_mesh_device();

    /**
     * @brief Setup fabric configuration before device initialization
     */
    void setup_fabric_configuration();

    /**
     * @brief Expand test configurations from TestConfig to ParsedTestConfig
     */
    void expand_test_configurations();

    /**
     * @brief Create MeshSockets for a given test configuration
     *
     * @param test The test configuration
     * @return std::vector<tt::tt_metal::distributed::MeshSocket> Vector of created sockets
     */
    std::vector<tt::tt_metal::distributed::MeshSocket> create_sockets_for_test(const ParsedTestConfig& test);

    /**
     * @brief Convert TestSocketConfig to tt::tt_metal::distributed::SocketConfig
     *
     * @param socket_config The YAML-parsed socket configuration
     * @param memory_config The memory configuration for the socket
     * @return tt::tt_metal::distributed::SocketConfig
     */
    tt::tt_metal::distributed::SocketConfig convert_to_socket_config(
        const TestSocketConfig& socket_config, const ParsedMemoryConfig& memory_config);

    /**
     * @brief Convert SocketConnectionConfig to tt::tt_metal::distributed::SocketConnection
     *
     * @param connection_config The YAML-parsed connection configuration
     * @return tt::tt_metal::distributed::SocketConnection
     */
    tt::tt_metal::distributed::SocketConnection convert_to_socket_connection(
        const SocketConnectionConfig& connection_config);

    /**
     * @brief Execute socket test using the test_socket_send_recv utility function
     *
     * @param socket The MeshSocket to test
     * @param test The test configuration containing memory and iteration settings
     */
    void execute_socket_test(tt::tt_metal::distributed::MeshSocket& socket, const ParsedTestConfig& test);

    /**
     * @brief Get the current distributed context
     *
     * @return std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>
     */
    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> get_distributed_context() const;

    /**
     * @brief Validate that the current rank can participate in the given test
     *
     * @param test The test configuration to validate
     * @return true if current rank should participate, false otherwise
     */
    bool should_participate_in_test(const ParsedTestConfig& test) const;

    /**
     * @brief Log test execution details
     *
     * @param test The test being executed
     * @param socket_index Index of the socket being tested
     * @param total_sockets Total number of sockets in the test
     */
    void log_test_execution(const ParsedTestConfig& test, size_t socket_index, size_t total_sockets) const;

    /**
     * @brief Create rank to mesh mapping using distributed allgather
     *
     * @return std::unordered_map<Rank, tt::tt_fabric::MeshId>
     */
    std::unordered_map<Rank, tt::tt_fabric::MeshId> create_rank_to_mesh_mapping();

    // Configuration and state
    MeshSocketTestConfiguration config_;
    std::vector<ParsedTestConfig> expanded_tests_;
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_;
    tt::tt_metal::distributed::multihost::Rank local_rank_;
    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> distributed_context_;

    // Test execution state
    std::unordered_map<std::string, size_t> test_name_to_index_;

    // Control plane and mesh configuration
    ControlPlane* control_plane_ptr_;
    MeshId local_mesh_id_;
    MeshShape mesh_shape_;
    std::unordered_map<Rank, tt::tt_fabric::MeshId> rank_to_mesh_mapping_;

    // Default test parameters
    static constexpr uint32_t DEFAULT_NUM_ITERATIONS = 1;
};

}  // namespace tt::tt_fabric::mesh_socket_tests
