// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <string>
#include <optional>
#include <unordered_map>

#include <yaml-cpp/yaml.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include "tests/tt_metal/test_utils/test_common.hpp"

namespace tt::tt_fabric::mesh_socket_tests {

using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;
using CoreCoord = tt::tt_metal::CoreCoord;
using Rank = tt::tt_metal::distributed::multihost::Rank;

enum class RoutingType : uint32_t {
    LowLatency = 0,
    Dynamic = 1,
};

/*  TODO: Add support for other patterns.
    Patterns need to split into three layers:
    1. host patterns (number of host pairs)
    2. device level patterns (number of sockets between each host pair)
    3. socket level patterns (number of connection in each socket)
    Currently patterns are only at the device level,
    going between all host pairs and one connection per socket. */
enum class PatternType : uint32_t {
    AllToAllDevices = 0,
    AllHostsRandomSockets = 1,
    AllDeviceBroadcast = 2,
};

// Data structures for parsed YAML configuration
struct PhysicalMeshConfig {
    std::string mesh_descriptor_path;
    std::vector<std::vector<eth_coord_t>> eth_coord_mapping;
};

struct FabricConfig {
    tt::tt_fabric::Topology topology;
    RoutingType routing_type;
};

struct MemoryConfig {
    std::vector<uint32_t> fifo_size;
    std::vector<uint32_t> page_size;
    std::vector<uint32_t> data_size;
    uint32_t num_transactions{};
};

struct ParsedMemoryConfig {
    uint32_t fifo_size;
    uint32_t page_size;
    uint32_t data_size;
    uint32_t num_transactions;
};

struct EndpointConfig {
    MeshCoordinate mesh_coord;
    CoreCoord core_coord;

    // Constructor
    EndpointConfig(const MeshCoordinate& mesh, const CoreCoord& core) : mesh_coord(mesh), core_coord(core) {}
};

struct SocketConnectionConfig {
    EndpointConfig sender;
    EndpointConfig receiver;
};

struct TestSocketConfig {
    std::vector<SocketConnectionConfig> connections;
    Rank sender_rank;
    Rank receiver_rank;
};

struct PatternExpansionConfig {
    PatternType type{};
    CoreCoord core_coord;  // Core coordinate to use for connections
    std::optional<uint32_t> num_sockets;  // Optional number of random sockets to generate
};

struct TestConfig {
    std::string name;
    std::optional<uint32_t> num_iterations;
    MemoryConfig memory_config;
    std::optional<std::vector<TestSocketConfig>> sockets;
    std::optional<std::vector<PatternExpansionConfig>> pattern_expansions;
};

// Second pass expansion of patterns
struct ParsedTestConfig {
    std::string name;
    std::optional<uint32_t> num_iterations;
    ParsedMemoryConfig memory_config;
    std::vector<TestSocketConfig> sockets;
};

struct MeshSocketTestConfiguration {
    std::optional<PhysicalMeshConfig> physical_mesh_config;
    FabricConfig fabric_config{};
    std::vector<TestConfig> tests;
};

class CmdlineParser {
public:
    CmdlineParser(const std::vector<std::string>& input_args) : input_args_(input_args) {}

    std::optional<std::string> get_yaml_config_path();
    bool has_help_option();
    void print_help();
    bool print_configs();

private:
    const std::vector<std::string>& input_args_;
};

// Forward declaration
class MeshSocketTestContext;

// Main YAML parser class
class MeshSocketYamlParser {
public:
    MeshSocketYamlParser() = delete;
    // Main parsing method
    static MeshSocketTestConfiguration parse_file(const std::string& yaml_file_path);

    // Two-stage parsing: YAML -> TestConfig -> ParsedTestConfig
    static std::vector<TestConfig> parse_raw_test_configs(const YAML::Node& tests_node);
    static std::vector<ParsedTestConfig> expand_test_configs(
        const std::vector<TestConfig>& test_configs, const MeshSocketTestContext& test_context);

    // Print configuration for debugging
    static void print_test_configuration(const MeshSocketTestConfiguration& config);

private:
    // Parsing helper methods
    static PhysicalMeshConfig parse_physical_mesh(const YAML::Node& node);
    static FabricConfig parse_fabric_config(const YAML::Node& node);
    static TestConfig parse_test_config(const YAML::Node& node);
    static TestSocketConfig parse_socket_config(const YAML::Node& node);
    static SocketConnectionConfig parse_connection_config(const YAML::Node& node);
    static EndpointConfig parse_endpoint_config(const YAML::Node& node);
    static MemoryConfig parse_memory_config(const YAML::Node& node);
    static PatternExpansionConfig parse_pattern_expansion(const YAML::Node& node);

    // Pattern expansion methods
    static std::vector<ParsedTestConfig> expand_test_config(
        const TestConfig& test_config, const MeshSocketTestContext& test_context);
    static std::vector<TestSocketConfig> expand_pattern(
        const PatternExpansionConfig& pattern, const MeshSocketTestContext& test_context);
    static std::vector<TestSocketConfig> expand_all_to_all_devices_pattern(
        const PatternExpansionConfig& pattern, const MeshSocketTestContext& test_context);
    static std::vector<TestSocketConfig> expand_all_hosts_random_sockets_pattern(
        const PatternExpansionConfig& pattern, const MeshSocketTestContext& test_context);
    static std::vector<TestSocketConfig> expand_all_device_broadcast_pattern(
        const PatternExpansionConfig& pattern, const MeshSocketTestContext& test_context);

    // Memory config expansion methods
    static std::vector<ParsedMemoryConfig> expand_memory_config(const MemoryConfig& memory_config);

    // Utility parsing methods
    static MeshCoordinate parse_mesh_coordinate(const YAML::Node& node);
    static CoreCoord parse_core_coordinate(const YAML::Node& node);
    static PatternType parse_pattern_type(const std::string& pattern_string);
    static std::vector<std::vector<eth_coord_t>> parse_eth_coord_mapping(const YAML::Node& node);

    // Validation helper methods
    static void validate_memory_config(const MemoryConfig& memory);
    static void validate_socket_config(
        const TestSocketConfig& socket_config, const MeshSocketTestContext& test_context);

    // Error handling
    static void throw_parse_error(const std::string& message, const YAML::Node& node = YAML::Node());
};

}  // namespace tt::tt_fabric::mesh_socket_tests
