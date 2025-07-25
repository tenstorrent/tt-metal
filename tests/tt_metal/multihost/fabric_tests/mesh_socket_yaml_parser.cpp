// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/multihost/fabric_tests/mesh_socket_yaml_parser.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>

#include <tt-logger/tt-logger.hpp>

namespace tt::tt_fabric::mesh_socket_tests {

MeshSocketTestConfiguration MeshSocketYamlParser::parse_file(const std::string& yaml_file_path) {
    log_info(tt::LogTest, "Parsing MeshSocket test configuration from: {}", yaml_file_path);

    std::ifstream file(yaml_file_path);
    TT_FATAL(file.is_open(), "Failed to open YAML configuration file: {}", yaml_file_path);

    YAML::Node root;
    try {
        root = YAML::LoadFile(yaml_file_path);
    } catch (const YAML::Exception& e) {
        TT_THROW("Failed to parse YAML file '{}': {}", yaml_file_path, e.what());
    }

    MeshSocketTestConfiguration config;

    // Parse optional physical mesh configuration
    if (root["physical_mesh"]) {
        config.physical_mesh_config = parse_physical_mesh(root["physical_mesh"]);
    }

    // Parse fabric configuration (required)
    TT_FATAL(root["fabric_config"], "Missing required 'fabric_config' section");
    config.fabric_config = parse_fabric_config(root["fabric_config"]);

    // Parse tests (required)
    TT_FATAL(root["tests"] && root["tests"].IsSequence(), "Missing or invalid 'tests' section - must be a list");

    // Two-stage parsing: YAML -> TestConfig -> ParsedTestConfig
    auto raw_test_configs = parse_raw_test_configs(root["tests"]);
    config.tests = expand_test_configs(raw_test_configs);

    // Validate the complete configuration
    validate_configuration(config);

    log_info(tt::LogTest, "Successfully parsed {} test configurations", config.tests.size());
    return config;
}

std::vector<TestConfig> MeshSocketYamlParser::parse_raw_test_configs(const YAML::Node& tests_node) {
    std::vector<TestConfig> test_configs;

    for (const auto& test_node : tests_node) {
        test_configs.push_back(parse_test_config(test_node));
    }

    return test_configs;
}

std::vector<ParsedTestConfig> MeshSocketYamlParser::expand_test_configs(const std::vector<TestConfig>& test_configs) {
    std::vector<ParsedTestConfig> parsed_configs;

    for (const auto& test_config : test_configs) {
        parsed_configs.push_back(expand_test_config(test_config));
    }

    return parsed_configs;
}

ParsedTestConfig MeshSocketYamlParser::expand_test_config(const TestConfig& test_config) {
    ParsedTestConfig parsed;
    parsed.name = test_config.name;
    parsed.num_iterations = test_config.num_iterations;
    parsed.memory_config = test_config.memory_config;

    // Validate that we have either explicit sockets or pattern expansions, but not both
    if (test_config.sockets.has_value() && test_config.pattern_expansions.has_value()) {
        TT_THROW("Test '{}' cannot have both explicit sockets and pattern expansions", test_config.name);
    }

    // Start with explicit sockets if they exist
    if (test_config.sockets.has_value()) {
        parsed.sockets = test_config.sockets.value();
    } else if (test_config.pattern_expansions.has_value()) {  // Expand patterns and add to sockets, cannot have both
        for (const auto& pattern : test_config.pattern_expansions.value()) {
            auto expanded_sockets = expand_pattern(pattern);
            parsed.sockets.insert(parsed.sockets.end(), expanded_sockets.begin(), expanded_sockets.end());
        }
    }

    // Validate the expanded test has at least one socket
    TT_FATAL(!parsed.sockets.empty(), "Test '{}' has no sockets", test_config.name);

    return parsed;
}

std::vector<SocketConfig> MeshSocketYamlParser::expand_pattern(const PatternExpansionConfig& pattern) {
    switch (pattern.type) {
        case PatternType::AllToAll: return expand_all_to_all_pattern(pattern);
        case PatternType::RandomPairing: return expand_random_pairing_pattern(pattern);
        default: TT_THROW("Unknown pattern type");
    }
}

std::vector<SocketConfig> MeshSocketYamlParser::expand_all_to_all_pattern(const PatternExpansionConfig& pattern) {
    std::vector<SocketConfig> sockets;

    // If pattern has explicit sockets defined, use those
    if (pattern.sockets.has_value()) {
        return pattern.sockets.value();
    }

    // TODO: Implement actual all-to-all expansion
    // This would require knowledge of available devices which isn't available at parse time
    // For now, return empty vector - this should be handled by the test runner
    log_warning(tt::LogTest, "All-to-all pattern expansion not implemented at parse time");

    return sockets;
}

std::vector<SocketConfig> MeshSocketYamlParser::expand_random_pairing_pattern(const PatternExpansionConfig& pattern) {
    std::vector<SocketConfig> sockets;

    // If pattern has explicit sockets defined, use those
    if (pattern.sockets.has_value()) {
        return pattern.sockets.value();
    }

    // TODO: Implement actual random pairing expansion
    // This would require knowledge of available devices which isn't available at parse time
    // For now, return empty vector - this should be handled by the test runner
    log_warning(tt::LogTest, "Random pairing pattern expansion not implemented at parse time");

    return sockets;
}

PhysicalMeshConfig MeshSocketYamlParser::parse_physical_mesh(const YAML::Node& node) {
    PhysicalMeshConfig config;

    if (node["mesh_descriptor_path"]) {
        config.mesh_descriptor_path = node["mesh_descriptor_path"].as<std::string>();
    }

    if (node["eth_coord_mapping"]) {
        config.eth_coord_mapping = parse_eth_coord_mapping(node["eth_coord_mapping"]);
    }

    return config;
}

FabricConfig MeshSocketYamlParser::parse_fabric_config(const YAML::Node& node) {
    FabricConfig config;

    TT_FATAL(node["topology"], "FabricConfig missing required 'topology' field");

    std::string topology_str = node["topology"].as<std::string>();

    if (topology_str == "Linear") {
        config.topology = tt::tt_fabric::Topology::Linear;
    } else if (topology_str == "Ring") {
        config.topology = tt::tt_fabric::Topology::Ring;
    } else if (topology_str == "Mesh") {
        config.topology = tt::tt_fabric::Topology::Mesh;
    } else {
        throw_parse_error("Invalid topology value: " + topology_str, node["topology"]);
    }

    if (node["routing_type"]) {
        std::string routing_str = node["routing_type"].as<std::string>();

        if (routing_str == "LowLatency") {
            config.routing_type = RoutingType::LowLatency;
        } else if (routing_str == "Dynamic") {
            config.routing_type = RoutingType::Dynamic;
        } else {
            throw_parse_error("Invalid routing_type value: " + routing_str, node["routing_type"]);
        }
    }

    return config;
}

TestConfig MeshSocketYamlParser::parse_test_config(const YAML::Node& node) {
    TestConfig test;

    TT_FATAL(node["name"], "Test configuration missing required 'name' field");
    test.name = node["name"].as<std::string>();

    if (node["num_iterations"]) {
        test.num_iterations = node["num_iterations"].as<uint32_t>();
    }

    TT_FATAL(node["memory_config"], "Test configuration missing required 'memory_config' field");
    test.memory_config = parse_memory_config(node["memory_config"]);

    // Parse explicit sockets (optional)
    if (node["sockets"]) {
        TT_FATAL(node["sockets"].IsSequence(), "'sockets' must be a list");
        std::vector<SocketConfig> sockets;
        for (const auto& socket_node : node["sockets"]) {
            sockets.push_back(parse_socket_config(socket_node));
        }
        test.sockets = sockets;
    }

    // Parse pattern expansions (optional)
    if (node["pattern_expansions"]) {
        TT_FATAL(node["pattern_expansions"].IsSequence(), "'pattern_expansions' must be a list");
        std::vector<PatternExpansionConfig> patterns;
        for (const auto& pattern_node : node["pattern_expansions"]) {
            patterns.push_back(parse_pattern_expansion(pattern_node));
        }
        test.pattern_expansions = patterns;
    }

    // Validate that test has either explicit sockets or pattern expansions
    if (!test.sockets.has_value() && !test.pattern_expansions.has_value()) {
        throw_parse_error("Test must define either 'sockets' or 'pattern_expansions'", node);
    }

    return test;
}

SocketConfig MeshSocketYamlParser::parse_socket_config(const YAML::Node& node) {
    SocketConfig socket;

    // Parse connections
    if (node["connections"]) {
        // Multi-connection socket
        TT_FATAL(node["connections"].IsSequence(), "'connections' must be a list");
        for (const auto& conn_node : node["connections"]) {
            socket.connections.push_back(parse_connection_config(conn_node));
        }
    } else if (node["sender"] && node["receiver"]) {
        // Single connection socket
        socket.connections.push_back(parse_connection_config(node));
    } else {
        throw_parse_error("Socket must define either 'connections' or 'sender'/'receiver' pair", node);
    }

    // Parse sender and receiver ranks (required)
    TT_FATAL(node["sender_rank"], "Socket missing required 'sender_rank' field");
    TT_FATAL(node["receiver_rank"], "Socket missing required 'receiver_rank' field");
    socket.sender_rank = Rank{node["sender_rank"].as<uint32_t>()};
    socket.receiver_rank = Rank{node["receiver_rank"].as<uint32_t>()};

    return socket;
}

SocketConnectionConfig MeshSocketYamlParser::parse_connection_config(const YAML::Node& node) {
    TT_FATAL(node["sender"], "Connection missing required 'sender' field");
    TT_FATAL(node["receiver"], "Connection missing required 'receiver' field");

    return SocketConnectionConfig{
        .sender = parse_endpoint_config(node["sender"]), .receiver = parse_endpoint_config(node["receiver"])};
}

EndpointConfig MeshSocketYamlParser::parse_endpoint_config(const YAML::Node& node) {
    TT_FATAL(node["mesh_coord"], "Endpoint missing required 'mesh_coord' field");
    TT_FATAL(node["core_coord"], "Endpoint missing required 'core_coord' field");

    return EndpointConfig{parse_mesh_coordinate(node["mesh_coord"]), parse_core_coordinate(node["core_coord"])};
}

MemoryConfig MeshSocketYamlParser::parse_memory_config(const YAML::Node& node) {
    MemoryConfig memory;

    if (node["fifo_size"]) {
        memory.fifo_size = node["fifo_size"].as<uint32_t>();
    }

    if (node["page_size"]) {
        memory.page_size = node["page_size"].as<uint32_t>();
    }

    if (node["data_size"]) {
        memory.data_size = node["data_size"].as<uint32_t>();
    }

    return memory;
}

PatternExpansionConfig MeshSocketYamlParser::parse_pattern_expansion(const YAML::Node& node) {
    PatternExpansionConfig pattern;

    TT_FATAL(node["type"], "Pattern expansion missing required 'type' field");

    std::string type_str = node["type"].as<std::string>();
    pattern.type = parse_pattern_type(type_str);

    return pattern;
}

MeshCoordinate MeshSocketYamlParser::parse_mesh_coordinate(const YAML::Node& node) {
    TT_FATAL(node.IsSequence() && node.size() == 2, "mesh_coord must be a 2-element array [row, col]");

    uint32_t row = node[0].as<uint32_t>();
    uint32_t col = node[1].as<uint32_t>();

    return MeshCoordinate(row, col);
}

CoreCoord MeshSocketYamlParser::parse_core_coordinate(const YAML::Node& node) {
    TT_FATAL(node.IsSequence() && node.size() == 2, "core_coord must be a 2-element array [x, y]");

    uint32_t x = node[0].as<uint32_t>();
    uint32_t y = node[1].as<uint32_t>();

    return CoreCoord(x, y);
}

PatternType MeshSocketYamlParser::parse_pattern_type(const std::string& pattern_string) {
    if (pattern_string == "all_to_all") {
        return PatternType::AllToAll;
    } else if (pattern_string == "random_pairing") {
        return PatternType::RandomPairing;
    } else {
        TT_THROW("Invalid pattern type: '{}'. Valid types are: all_to_all, random_pairing", pattern_string);
    }
}

std::vector<std::vector<eth_coord_t>> MeshSocketYamlParser::parse_eth_coord_mapping(const YAML::Node& yaml_node) {
    std::vector<std::vector<eth_coord_t>> array;

    TT_FATAL(yaml_node.IsSequence(), "Expected a sequence for 2D array");

    for (const auto& row : yaml_node) {
        TT_FATAL(row.IsSequence(), "Expected each row to be a sequence");
        std::vector<eth_coord_t> row_vector;
        row_vector.reserve(row.size());
        for (const auto& entry : row) {
            TT_FATAL(entry.size() == 5, "Expected ethernet core coordinates to be a sequence of 5 elements");
            row_vector.push_back(eth_coord_t{
                entry[0].as<uint32_t>(),
                entry[1].as<uint32_t>(),
                entry[2].as<uint32_t>(),
                entry[3].as<uint32_t>(),
                entry[4].as<uint32_t>()});
        }
        array.push_back(std::move(row_vector));
    }

    return array;
}

void MeshSocketYamlParser::validate_configuration(const MeshSocketTestConfiguration& config) {
    // Validate that we have at least one test
    TT_FATAL(!config.tests.empty(), "Configuration must define at least one test");

    // Validate fabric config
    validate_fabric_config(config.fabric_config);

    // Validate physical mesh config if present
    if (config.physical_mesh_config.has_value()) {
        validate_physical_mesh_config(config.physical_mesh_config.value());
    }

    // Validate each parsed test
    for (const auto& test : config.tests) {
        validate_parsed_test_config(test);
    }

    log_info(tt::LogTest, "Configuration validation passed");
}

void MeshSocketYamlParser::validate_test_config(const TestConfig& test) {
    TT_FATAL(!test.name.empty(), "Test name cannot be empty");

    // Validate memory config
    validate_memory_config(test.memory_config);

    // Validate explicit sockets if present
    if (test.sockets.has_value()) {
        for (const auto& socket : test.sockets.value()) {
            validate_socket_config(socket);
        }
    }

    // Validate pattern expansions if present
    if (test.pattern_expansions.has_value()) {
        for (const auto& pattern : test.pattern_expansions.value()) {
            validate_pattern_expansion_config(pattern);
        }
    }
}

void MeshSocketYamlParser::validate_parsed_test_config(const ParsedTestConfig& test) {
    TT_FATAL(!test.name.empty(), "Test name cannot be empty");

    TT_FATAL(!test.sockets.empty(), "Parsed test '{}' must have at least one socket", test.name);

    // Validate memory config
    validate_memory_config(test.memory_config);

    // Validate all sockets
    for (const auto& socket : test.sockets) {
        validate_socket_config(socket);
    }
}

void MeshSocketYamlParser::validate_socket_config(const SocketConfig& socket) {
    TT_FATAL(!socket.connections.empty(), "Socket must have at least one connection");

    for (const auto& connection : socket.connections) {
        validate_endpoint_config(connection.sender);
        validate_endpoint_config(connection.receiver);
    }
}

void MeshSocketYamlParser::validate_memory_config(const MemoryConfig& memory) {
    TT_FATAL(memory.fifo_size > 0, "fifo_size must be greater than 0");
    TT_FATAL(memory.page_size > 0, "page_size must be greater than 0");
    TT_FATAL(memory.data_size > 0, "data_size must be greater than 0");
}

void MeshSocketYamlParser::validate_endpoint_config(const EndpointConfig& endpoint) {
    // Basic validation - more detailed validation will happen at runtime
    // when we know the actual mesh dimensions
    (void)endpoint;  // Suppress unused parameter warning
}

void MeshSocketYamlParser::validate_pattern_expansion_config(const PatternExpansionConfig& pattern) {
    // Pattern type is validated during parsing via enum
    // Additional validation could be added here for specific pattern requirements
    (void)pattern;  // Suppress unused parameter warning
}

void MeshSocketYamlParser::validate_fabric_config(const FabricConfig& fabric_config) {
    // Topology is validated during parsing via enum
    // Routing type is optional and validated during parsing if present
    (void)fabric_config;  // Suppress unused parameter warning
}

void MeshSocketYamlParser::validate_physical_mesh_config(const PhysicalMeshConfig& physical_mesh_config) {
    if (!physical_mesh_config.mesh_descriptor_path.empty()) {
        // Could add file existence check here
        log_debug(tt::LogTest, "Physical mesh descriptor path: {}", physical_mesh_config.mesh_descriptor_path);
    }

    if (!physical_mesh_config.eth_coord_mapping.empty()) {
        log_debug(
            tt::LogTest, "Ethernet coordinate mapping has {} rows", physical_mesh_config.eth_coord_mapping.size());
    }
}

void MeshSocketYamlParser::throw_parse_error(const std::string& message, const YAML::Node& node) {
    std::string error_msg = "YAML Parse Error: " + message;

    if (node.IsDefined()) {
        error_msg += " (line " + std::to_string(node.Mark().line + 1) + ", column " +
                     std::to_string(node.Mark().column + 1) + ")";
    }

    TT_THROW("{}", error_msg);
}

}  // namespace tt::tt_fabric::mesh_socket_tests
