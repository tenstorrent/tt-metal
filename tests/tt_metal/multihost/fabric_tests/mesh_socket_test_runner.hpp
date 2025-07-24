// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <random>
#include <unordered_map>

#include "tests/tt_metal/multihost/fabric_tests/mesh_socket_yaml_parser.hpp"
#include "tests/tt_metal/multihost/fabric_tests/socket_send_recv_utils.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/fabric.hpp>

namespace tt::tt_fabric::mesh_socket_tests {

using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using MeshSocket = tt::tt_metal::distributed::MeshSocket;
using SocketConfig = tt::tt_metal::distributed::SocketConfig;
using SocketConnection = tt::tt_metal::distributed::SocketConnection;
using SocketMemoryConfig = tt::tt_metal::distributed::SocketMemoryConfig;
using DistributedContext = tt::tt_metal::distributed::multihost::DistributedContext;

// Additional data structures for test execution
struct ResolvedSocketConfig {
    SocketConfig tt_socket_config;
    MemoryConfig memory;
    uint32_t num_iterations;
};

struct ResolvedTestConfig {
    std::string name;
    std::string description;
    std::vector<ResolvedSocketConfig> sockets;
};

struct TestExecutionContext {
    std::shared_ptr<MeshDevice> mesh_device;
    uint32_t current_rank;
    uint32_t total_ranks;
    std::shared_ptr<DistributedContext> distributed_context;
};

struct TestExecutionStats {
    uint32_t tests_run = 0;
    uint32_t tests_passed = 0;
    uint32_t tests_failed = 0;
    uint32_t sockets_tested = 0;
    std::chrono::milliseconds total_execution_time{0};
};

// Main test runner class
class MeshSocketTestRunner {
public:
    MeshSocketTestRunner();
    ~MeshSocketTestRunner();

    // Main execution methods
    void run_tests_from_config(const MeshSocketTestConfiguration& config);
    void setup_fabric_context(const std::optional<PhysicalMeshConfig>& mesh_config);
    void cleanup();

    // Statistics and reporting
    const TestExecutionStats& get_execution_stats() const { return stats_; }
    void print_execution_summary() const;

private:
    // Test generation methods
    std::vector<ResolvedTestConfig> generate_tests(const MeshSocketTestConfiguration& config);
    ResolvedTestConfig resolve_test_config(const TestConfig& test, const DefaultConfig& defaults);
    std::vector<ResolvedSocketConfig> expand_pattern_to_sockets(
        const PatternExpansionConfig& pattern, const MemoryConfig& memory, uint32_t iterations);

    // Pattern expansion methods
    std::vector<ResolvedSocketConfig> expand_all_to_all_devices(
        const PatternExpansionConfig& pattern, const MemoryConfig& memory, uint32_t iterations);
    std::vector<ResolvedSocketConfig> expand_all_to_all_meshes(
        const PatternExpansionConfig& pattern, const MemoryConfig& memory, uint32_t iterations);
    std::vector<ResolvedSocketConfig> expand_random_pairings(
        const PatternExpansionConfig& pattern, const MemoryConfig& memory, uint32_t iterations);

    // Socket configuration methods
    ResolvedSocketConfig create_resolved_socket_config(
        const SocketConfig& socket, const MemoryConfig& default_memory, uint32_t default_iterations);
    SocketConfig create_tt_socket_config(const SocketConnectionConfig& connection, const MemoryConfig& memory);
    SocketMemoryConfig create_socket_memory_config(const MemoryConfig& memory);
    std::vector<SocketConnection> create_socket_connections(const std::vector<SocketConnectionConfig>& connections);

    // Test execution methods
    void execute_test(const ResolvedTestConfig& test);
    void execute_socket_test(const ResolvedSocketConfig& socket);

    // Host communication and synchronization
    void setup_host_communication();
    uint32_t exchange_seed_with_hosts();
    void synchronize_hosts();

    // Validation methods
    void validate_mesh_coordinates(const std::vector<MeshCoordinate>& coords);
    void validate_socket_configuration(const ResolvedSocketConfig& socket);
    void validate_runtime_environment();

    // Device discovery and management
    std::vector<MeshCoordinate> discover_available_devices();
    std::vector<uint32_t> get_available_host_ranks();
    bool is_sender_for_socket(const ResolvedSocketConfig& socket) const;
    bool is_receiver_for_socket(const ResolvedSocketConfig& socket) const;

    // Utility methods
    MemoryConfig merge_memory_config(const std::optional<MemoryConfig>& specific, const MemoryConfig& default_config);
    bool should_run_test(const std::string& test_name, const std::optional<std::vector<std::string>>& run_tests);
    void log_test_progress(const std::string& test_name, size_t current, size_t total);
    void record_test_result(const std::string& test_name, bool success, std::chrono::milliseconds execution_time);

    // Random number generation for pattern expansion
    uint32_t get_random_device_index();
    std::pair<MeshCoordinate, MeshCoordinate> get_random_device_pair();

    // Error handling and recovery
    void handle_test_failure(const std::string& test_name, const std::exception& e);
    void cleanup_failed_test();

private:
    TestExecutionContext execution_context_;
    TestExecutionStats stats_;
    std::mt19937 random_generator_;

    // Cached device and mesh information
    std::vector<MeshCoordinate> available_devices_;
    std::vector<uint32_t> available_host_ranks_;
    MeshShape mesh_shape_;

    // Configuration state
    bool is_initialized_ = false;
    std::string current_test_name_;

    // Constants
    static constexpr uint32_t DEFAULT_RANDOM_SEED = 12345;
    static constexpr uint32_t MAX_RETRY_ATTEMPTS = 3;
};

}  // namespace tt::tt_fabric::mesh_socket_tests
