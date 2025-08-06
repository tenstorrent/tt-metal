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
#include "umd/device/types/cluster_descriptor_types.h"
#include "tests/tt_metal/test_utils/test_common.hpp"

namespace tt::tt_fabric::mesh_socket_tests {

using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;
using CoreCoord = tt::tt_metal::CoreCoord;
using Rank = tt::tt_metal::distributed::multihost::Rank;

enum class RoutingType : uint32_t {
    LowLatency = 0,
    Dynamic = 1,
};

enum class PatternType {
    AllToAll,
    RandomPairing,
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
    std::vector<uint32_t> num_transactions;
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

// TODO: remove this and just have optional vector of PatternType instead in TestConfig. Do not do this unless I am
// sure.
struct PatternExpansionConfig {
    PatternType type;  // "all_to_all" or "random_pairing"
    CoreCoord core_coord;  // Core coordinate to use for connections
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
    FabricConfig fabric_config;
    std::vector<TestConfig> tests;
};

class CmdlineParser {
public:
    CmdlineParser(const std::vector<std::string>& input_args) : input_args_(input_args) {}

    std::optional<std::string> get_yaml_config_path();
    std::optional<uint32_t> get_master_seed();
    bool has_help_option();
    void print_help();
    bool print_configs();

private:
    const std::vector<std::string>& input_args_;
};

// Forward declaration
class MeshSocketTestRunner;

// Main YAML parser class
class MeshSocketYamlParser {
public:
    MeshSocketYamlParser() = default;
    ~MeshSocketYamlParser() = default;

    // Main parsing method
    MeshSocketTestConfiguration parse_file(const std::string& yaml_file_path);

    // Two-stage parsing: YAML -> TestConfig -> ParsedTestConfig
    std::vector<TestConfig> parse_raw_test_configs(const YAML::Node& tests_node);
    std::vector<ParsedTestConfig> expand_test_configs(
        const std::vector<TestConfig>& test_configs, const MeshSocketTestRunner& test_runner);

    // Validation methods
    static void validate_configuration(const MeshSocketTestConfiguration& config);

    // Print configuration for debugging
    static void print_test_configuration(const MeshSocketTestConfiguration& config);

private:
    // Parsing helper methods
    PhysicalMeshConfig parse_physical_mesh(const YAML::Node& node);
    FabricConfig parse_fabric_config(const YAML::Node& node);
    TestConfig parse_test_config(const YAML::Node& node);
    TestSocketConfig parse_socket_config(const YAML::Node& node);
    SocketConnectionConfig parse_connection_config(const YAML::Node& node);
    EndpointConfig parse_endpoint_config(const YAML::Node& node);
    MemoryConfig parse_memory_config(const YAML::Node& node);
    PatternExpansionConfig parse_pattern_expansion(const YAML::Node& node);

    // Pattern expansion methods
    std::vector<ParsedTestConfig> expand_test_config(
        const TestConfig& test_config, const MeshSocketTestRunner& test_runner);
    std::vector<TestSocketConfig> expand_pattern(
        const PatternExpansionConfig& pattern, const TestConfig& test_config, const MeshSocketTestRunner& test_runner);
    std::vector<TestSocketConfig> expand_all_to_all_pattern(
        const PatternExpansionConfig& pattern, const TestConfig& test_config, const MeshSocketTestRunner& test_runner);
    std::vector<TestSocketConfig> expand_random_pairing_pattern(
        const PatternExpansionConfig& pattern, const TestConfig& test_config, const MeshSocketTestRunner& test_runner);

    // Memory config expansion methods
    std::vector<ParsedMemoryConfig> expand_memory_config(const MemoryConfig& memory_config);

    // Utility parsing methods
    MeshCoordinate parse_mesh_coordinate(const YAML::Node& node);
    CoreCoord parse_core_coordinate(const YAML::Node& node);
    PatternType parse_pattern_type(const std::string& pattern_string);
    std::vector<std::vector<eth_coord_t>> parse_eth_coord_mapping(const YAML::Node& node);

    // Validation helper methods
    static void validate_test_config(const TestConfig& test);
    static void validate_parsed_test_config(const ParsedTestConfig& test);
    static void validate_socket_config(const TestSocketConfig& socket);
    static void validate_memory_config(const MemoryConfig& memory);
    static void validate_parsed_memory_config(const ParsedMemoryConfig& memory);
    static void validate_endpoint_config(const EndpointConfig& endpoint);
    static void validate_pattern_expansion_config(const PatternExpansionConfig& pattern);
    static void validate_fabric_config(const FabricConfig& fabric_config);
    static void validate_physical_mesh_config(const PhysicalMeshConfig& physical_mesh_config);

    // Error handling
    void throw_parse_error(const std::string& message, const YAML::Node& node = YAML::Node());
};

}  // namespace tt::tt_fabric::mesh_socket_tests
