// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <set>

#include "tests/tt_metal/multihost/fabric_tests/mesh_socket_yaml_parser.hpp"
#include "tests/tt_metal/multihost/fabric_tests/mesh_socket_test_context.hpp"

#include <tt-logger/tt-logger.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include <enchantum/enchantum.hpp>
#include <tt_stl/tt_stl/caseless_comparison.hpp>

namespace tt::tt_fabric::mesh_socket_tests {

std::optional<std::string> CmdlineParser::get_yaml_config_path() {
    std::string yaml_config = test_args::get_command_option(input_args_, "--test_config", "");

    if (!yaml_config.empty()) {
        std::filesystem::path fpath(yaml_config);
        if (!fpath.is_absolute()) {
            const auto& fname = fpath.filename();
            fpath = std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
                    "tests/tt_metal/multihost/fabric_tests/" / fname;
            log_warning(tt::LogTest, "Relative fpath for config provided, using absolute path: {}", fpath);
        }
        return fpath.string();
    }

    return std::nullopt;
}

bool CmdlineParser::has_help_option() { return test_args::has_command_option(input_args_, "--help"); }

bool CmdlineParser::print_configs() { return test_args::has_command_option(input_args_, "--print-configs"); }

void CmdlineParser::print_help() {
    log_info(tt::LogTest, "Usage: mesh_socket_test_main --test_config FILE");
    log_info(tt::LogTest, "");
    log_info(tt::LogTest, "--test_config FILE     Path to the YAML test configuration file. ");
    log_info(tt::LogTest, "--master-seed SEED     Master seed for all random operations to ensure reproducibility.");
    log_info(tt::LogTest, "--print-configs        Print the parsed test configurations and exit.");
    log_info(tt::LogTest, "--help                 Print this help message.");
    log_info(tt::LogTest, "");
    log_info(
        tt::LogTest,
        "Example: mesh_socket_test_main --test_config "
        "tests/tt_metal/multihost/fabric_tests/mesh_socket_test_config_example.yaml --master-seed 12345");
}

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
    if (root["physical_mesh"].IsDefined()) {
        config.physical_mesh_config = parse_physical_mesh(root["physical_mesh"]);
    }

    // Parse fabric configuration (required)
    TT_FATAL(root["fabric_config"].IsDefined(), "Missing required 'fabric_config' section");
    config.fabric_config = parse_fabric_config(root["fabric_config"]);

    // Parse tests (required)
    TT_FATAL(
        root["tests"].IsDefined() && root["tests"].IsSequence(), "Missing or invalid 'tests' section - must be a list");

    // Parse YAML -> TestConfig (no expansion at parse time)
    config.tests = parse_raw_test_configs(root["tests"]);

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

std::vector<ParsedTestConfig> MeshSocketYamlParser::expand_test_configs(
    const std::vector<TestConfig>& test_configs, const MeshSocketTestContext& test_context) {
    std::vector<ParsedTestConfig> parsed_configs;

    for (const auto& test_config : test_configs) {
        auto expanded_configs = expand_test_config(test_config, test_context);
        parsed_configs.insert(parsed_configs.end(), expanded_configs.begin(), expanded_configs.end());
    }

    return parsed_configs;
}

std::vector<ParsedTestConfig> MeshSocketYamlParser::expand_test_config(
    const TestConfig& test_config, const MeshSocketTestContext& test_context) {
    std::vector<ParsedTestConfig> parsed_configs;

    // Validate that we have either explicit sockets or pattern expansions, but not both
    if (test_config.sockets.has_value() && test_config.pattern_expansions.has_value()) {
        TT_THROW("Test '{}' cannot have both explicit sockets and pattern expansions", test_config.name);
    }

    // Expand memory configurations into all combinations
    auto expanded_memory_configs = expand_memory_config(test_config.memory_config);
    log_info(
        tt::LogTest,
        "Test '{}': Expanded memory config into {} combinations",
        test_config.name,
        expanded_memory_configs.size());

    // For each memory configuration, create test configurations
    for (size_t mem_idx = 0; mem_idx < expanded_memory_configs.size(); ++mem_idx) {
        const auto& memory_config = expanded_memory_configs[mem_idx];

        // Generate test name suffix for parameterized tests
        std::string test_name = test_config.name;
        if (expanded_memory_configs.size() > 1) {
            test_name += "_mem_" + std::to_string(mem_idx);
        }

        // Start with explicit sockets if they exist
        if (test_config.sockets.has_value()) {
            ParsedTestConfig parsed_config{
                .name = test_name,
                .num_iterations = test_config.num_iterations,
                .memory_config = memory_config,
                .sockets = test_config.sockets.value()};

            // Validate each socket configuration
            for (const auto& socket_config : parsed_config.sockets) {
                validate_socket_config(socket_config, test_context);
            }

            parsed_configs.emplace_back(std::move(parsed_config));
        } else if (test_config.pattern_expansions.has_value()) {
            for (size_t i = 0; i < test_config.pattern_expansions.value().size(); ++i) {
                const auto& pattern = test_config.pattern_expansions.value()[i];
                auto expanded_sockets = expand_pattern(pattern, test_context);
                test_name += "_pattern_" + std::to_string(i);
                parsed_configs.emplace_back(ParsedTestConfig{
                    .name = test_name,
                    .num_iterations = test_config.num_iterations,
                    .memory_config = memory_config,
                    .sockets = expanded_sockets});
            }
        }
    }

    // Validate the expanded test has at least one socket
    for (const auto& parsed_config : parsed_configs) {
        TT_FATAL(!parsed_config.sockets.empty(), "Test '{}' has no sockets", parsed_config.name);
    }

    return parsed_configs;
}

std::vector<TestSocketConfig> MeshSocketYamlParser::expand_pattern(

    const PatternExpansionConfig& pattern, const MeshSocketTestContext& test_context) {
    switch (pattern.type) {
        case PatternType::AllToAllDevices: return expand_all_to_all_devices_pattern(pattern, test_context);
        case PatternType::AllHostsRandomSockets: return expand_all_hosts_random_sockets_pattern(pattern, test_context);
        case PatternType::AllDeviceBroadcast: return expand_all_device_broadcast_pattern(pattern, test_context);
        default: TT_THROW("Unknown pattern type");
    }
}

// Parametrize on fifo size, page size, and data size

std::vector<TestSocketConfig> MeshSocketYamlParser::expand_all_to_all_devices_pattern(
    const PatternExpansionConfig& pattern, const MeshSocketTestContext& test_context) {
    std::vector<TestSocketConfig> sockets;

    const auto& mesh_graph = test_context.get_mesh_graph();
    const auto& rank_to_mesh_id = test_context.get_rank_to_mesh_mapping();
    const CoreCoord& core_coord = pattern.core_coord;

    // Create all-to-all connections between ranks
    for (const auto& [sender_rank, sender_mesh_id] : rank_to_mesh_id) {
        for (const auto& [receiver_rank, recv_mesh_id] : rank_to_mesh_id) {
            if (sender_rank == receiver_rank) {
                continue;  // Skip self-connections
            }

            auto sender_coord_range = mesh_graph.get_coord_range(sender_mesh_id);
            auto recv_coord_range = mesh_graph.get_coord_range(recv_mesh_id);
            // Create connections for each coordinate in the mesh
            for (const auto& sender_coord : sender_coord_range) {
                for (const auto& recv_coord : recv_coord_range) {
                    std::vector<SocketConnectionConfig> connections;
                    connections.push_back(SocketConnectionConfig{
                        .sender = EndpointConfig(sender_coord, core_coord),
                        .receiver = EndpointConfig(recv_coord, core_coord)});
                    sockets.push_back(TestSocketConfig{
                        .connections = connections,
                        .sender_rank = Rank{*sender_rank},
                        .receiver_rank = Rank{*receiver_rank}});
                }
            }
        }
    }

    log_info(tt::LogTest, "Generated {} sockets for all_to_all_device_unicast pattern", sockets.size());
    return sockets;
}

/* Create num_sockets between each rank pair, with random sender and receiver devices
 *  The purpose of this pattern is to reduce the number of sockets compared to all-to-all,
 *  while still ensuring that all ranks are connected to each other.
 */
std::vector<TestSocketConfig> MeshSocketYamlParser::expand_all_hosts_random_sockets_pattern(
    const PatternExpansionConfig& pattern, const MeshSocketTestContext& test_context) {
    std::vector<TestSocketConfig> sockets;

    const auto& mesh_graph = test_context.get_mesh_graph();
    const auto& rank_to_mesh_id = test_context.get_rank_to_mesh_mapping();
    const CoreCoord& core_coord = pattern.core_coord;

    uint32_t num_sockets = pattern.num_sockets.value();

    auto rng = test_context.get_rng();

    for (const auto& [sender_rank, sender_mesh_id] : rank_to_mesh_id) {
        for (const auto& [receiver_rank, recv_mesh_id] : rank_to_mesh_id) {
            if (sender_rank == receiver_rank) {
                continue;  // Skip self-connections
            }
            auto num_sender_devices = mesh_graph.get_mesh_shape(sender_mesh_id).mesh_size();
            auto num_recv_devices = mesh_graph.get_mesh_shape(recv_mesh_id).mesh_size();
            for (uint32_t i = 0; i < num_sockets; ++i) {
                auto sender_idx = rng() % num_sender_devices;
                auto recv_idx = rng() % num_recv_devices;
                std::vector<SocketConnectionConfig> connections;
                connections.push_back(SocketConnectionConfig{
                    .sender = EndpointConfig(mesh_graph.chip_to_coordinate(sender_mesh_id, sender_idx), core_coord),
                    .receiver = EndpointConfig(mesh_graph.chip_to_coordinate(recv_mesh_id, recv_idx), core_coord)});
                sockets.push_back(TestSocketConfig{
                    .connections = connections, .sender_rank = sender_rank, .receiver_rank = receiver_rank});
                log_info(
                    tt::LogTest,
                    "Generated socket: {} {} -> {} {}",
                    *sender_rank,
                    mesh_graph.chip_to_coordinate(sender_mesh_id, sender_idx),
                    *receiver_rank,
                    mesh_graph.chip_to_coordinate(recv_mesh_id, recv_idx));
            }
        }
    }

    log_info(tt::LogTest, "Generated {} sockets for all_hosts_random_sockets pattern", sockets.size());
    return sockets;
}

/* Create broadcast sockets where each device on every host acts as a sender to all devices on other hosts.
 * For each device coordinate on each rank, create sockets to all other ranks where the sender is that
 * specific device and the receivers are all devices on the target rank.
 */
std::vector<TestSocketConfig> MeshSocketYamlParser::expand_all_device_broadcast_pattern(
    const PatternExpansionConfig& pattern, const MeshSocketTestContext& test_context) {
    std::vector<TestSocketConfig> sockets;

    const auto& mesh_graph = test_context.get_mesh_graph();
    const auto& rank_to_mesh_id = test_context.get_rank_to_mesh_mapping();
    const CoreCoord& core_coord = pattern.core_coord;

    for (const auto& [sender_rank, sender_mesh_id] : rank_to_mesh_id) {
        auto sender_coord_range = mesh_graph.get_coord_range(sender_mesh_id);
        for (const auto& sender_coord : sender_coord_range) {
            for (const auto& [receiver_rank, recv_mesh_id] : rank_to_mesh_id) {
                if (sender_rank == receiver_rank) {
                    continue;  // Skip self-connections
                }
                auto recv_coord_range = mesh_graph.get_coord_range(recv_mesh_id);
                std::vector<SocketConnectionConfig> connections;

                for (const auto& recv_coord : recv_coord_range) {
                    connections.push_back(SocketConnectionConfig{
                        .sender = EndpointConfig(sender_coord, core_coord),
                        .receiver = EndpointConfig(recv_coord, core_coord)});
                }

                sockets.push_back(TestSocketConfig{
                    .connections = connections,
                    .sender_rank = Rank{*sender_rank},
                    .receiver_rank = Rank{*receiver_rank}});
                log_info(
                    tt::LogTest,
                    "Generated broadcast socket: device {} on rank {} -> all {} devices on rank {}",
                    sender_coord,
                    *sender_rank,
                    connections.size(),
                    *receiver_rank);
            }
        }
    }

    log_info(tt::LogTest, "Generated {} sockets for all_device_broadcast pattern", sockets.size());
    return sockets;
}

std::vector<ParsedMemoryConfig> MeshSocketYamlParser::expand_memory_config(const MemoryConfig& memory_config) {
    std::vector<ParsedMemoryConfig> expanded_configs;

    for (uint32_t fifo_size : memory_config.fifo_size) {
        for (uint32_t page_size : memory_config.page_size) {
            for (uint32_t data_size : memory_config.data_size) {
                expanded_configs.push_back(ParsedMemoryConfig{
                    .fifo_size = fifo_size,
                    .page_size = page_size,
                    .data_size = data_size,
                    .num_transactions = memory_config.num_transactions});
            }
        }
    }

    return expanded_configs;
}

PhysicalMeshConfig MeshSocketYamlParser::parse_physical_mesh(const YAML::Node& node) {
    PhysicalMeshConfig config;
    TT_FATAL(
        node["mesh_descriptor_path"].IsDefined() && node["eth_coord_mapping"].IsDefined(),
        "PhysicalMeshConfig missing required 'mesh_descriptor_path' or 'eth_coord_mapping' field");

    config.mesh_descriptor_path = node["mesh_descriptor_path"].as<std::string>();
    config.eth_coord_mapping = parse_eth_coord_mapping(node["eth_coord_mapping"]);
    return config;
}

FabricConfig MeshSocketYamlParser::parse_fabric_config(const YAML::Node& node) {
    FabricConfig config{};

    TT_FATAL(node["topology"].IsDefined(), "FabricConfig missing required 'topology' field");

    std::string topology_str = node["topology"].as<std::string>();

    auto topology = enchantum::cast<tt::tt_fabric::Topology>(topology_str, ttsl::ascii_caseless_comp);
    if (topology.has_value()) {
        config.topology = topology.value();
    } else {
        throw_parse_error("Invalid topology value: " + topology_str, node["topology"]);
    }

    return config;
}

TestConfig MeshSocketYamlParser::parse_test_config(const YAML::Node& node) {
    TestConfig test;

    TT_FATAL(node["name"].IsDefined(), "Test configuration missing required 'name' field");
    test.name = node["name"].as<std::string>();
    TT_FATAL(!test.name.empty(), "Test name cannot be empty");

    if (node["num_iterations"].IsDefined()) {
        test.num_iterations = node["num_iterations"].as<uint32_t>();
    }

    TT_FATAL(node["memory_config"].IsDefined(), "Test configuration missing required 'memory_config' field");
    test.memory_config = parse_memory_config(node["memory_config"]);
    validate_memory_config(test.memory_config);

    if (node["sockets"].IsDefined()) {
        TT_FATAL(node["sockets"].IsSequence(), "'sockets' must be a list");
        std::vector<TestSocketConfig> sockets;
        for (const auto& socket_node : node["sockets"]) {
            sockets.push_back(parse_socket_config(socket_node));
        }
        test.sockets = sockets;
    }

    if (node["pattern_expansions"].IsDefined()) {
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

TestSocketConfig MeshSocketYamlParser::parse_socket_config(const YAML::Node& node) {
    TestSocketConfig socket;

    if (node["connections"].IsDefined()) {
        TT_FATAL(node["connections"].IsSequence(), "'connections' must be a list");
        for (const auto& conn_node : node["connections"]) {
            socket.connections.push_back(parse_connection_config(conn_node));
        }
    } else {
        throw_parse_error("Socket must define either 'connections'", node);
    }

    TT_FATAL(node["sender_rank"].IsDefined(), "Socket missing required 'sender_rank' field");
    TT_FATAL(node["receiver_rank"].IsDefined(), "Socket missing required 'receiver_rank' field");
    socket.sender_rank = Rank{node["sender_rank"].as<uint32_t>()};
    socket.receiver_rank = Rank{node["receiver_rank"].as<uint32_t>()};
    TT_FATAL(socket.sender_rank != socket.receiver_rank, "Sender and receiver ranks must be different");

    return socket;
}

SocketConnectionConfig MeshSocketYamlParser::parse_connection_config(const YAML::Node& node) {
    TT_FATAL(node["sender"].IsDefined(), "Connection missing required 'sender' field");
    TT_FATAL(node["receiver"].IsDefined(), "Connection missing required 'receiver' field");

    return SocketConnectionConfig{
        .sender = parse_endpoint_config(node["sender"]), .receiver = parse_endpoint_config(node["receiver"])};
}

EndpointConfig MeshSocketYamlParser::parse_endpoint_config(const YAML::Node& node) {
    TT_FATAL(node["mesh_coord"].IsDefined(), "Endpoint missing required 'mesh_coord' field");
    TT_FATAL(node["core_coord"].IsDefined(), "Endpoint missing required 'core_coord' field");

    return EndpointConfig{parse_mesh_coordinate(node["mesh_coord"]), parse_core_coordinate(node["core_coord"])};
}

MemoryConfig MeshSocketYamlParser::parse_memory_config(const YAML::Node& node) {
    MemoryConfig memory;

    if (node["fifo_size"].IsDefined()) {
        if (node["fifo_size"].IsSequence()) {
            memory.fifo_size = node["fifo_size"].as<std::vector<uint32_t>>();
        } else {
            memory.fifo_size = {node["fifo_size"].as<uint32_t>()};
        }
    }

    if (node["page_size"].IsDefined()) {
        if (node["page_size"].IsSequence()) {
            memory.page_size = node["page_size"].as<std::vector<uint32_t>>();
        } else {
            memory.page_size = {node["page_size"].as<uint32_t>()};
        }
    }
    if (node["data_size"].IsDefined()) {
        if (node["data_size"].IsSequence()) {
            memory.data_size = node["data_size"].as<std::vector<uint32_t>>();
        } else {
            memory.data_size = {node["data_size"].as<uint32_t>()};
        }
    }

    if (node["num_transactions"].IsDefined()) {
        memory.num_transactions = node["num_transactions"].as<uint32_t>();
    }

    return memory;
}

PatternExpansionConfig MeshSocketYamlParser::parse_pattern_expansion(const YAML::Node& node) {
    PatternExpansionConfig pattern;

    TT_FATAL(node["type"].IsDefined(), "Pattern expansion missing required 'type' field");

    pattern.type = parse_pattern_type(node["type"].as<std::string>());

    // Parse core_coord field (required)
    TT_FATAL(node["core_coord"].IsDefined(), "Pattern expansion missing required 'core_coord' field");
    pattern.core_coord = parse_core_coordinate(node["core_coord"]);

    // Optional num_sockets
    if (node["num_sockets"].IsDefined()) {
        pattern.num_sockets = node["num_sockets"].as<uint32_t>();
    }

    TT_FATAL(
        pattern.type == PatternType::AllHostsRandomSockets ? pattern.num_sockets.has_value() : true,
        "num_sockets must be defined for all_hosts_random_sockets pattern expansions");

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
    if (pattern_string == "all_to_all_device_unicast") {
        return PatternType::AllToAllDevices;
    }
    if (pattern_string == "all_hosts_random_sockets") {
        return PatternType::AllHostsRandomSockets;
    }
    if (pattern_string == "all_device_broadcast") {
        return PatternType::AllDeviceBroadcast;
    }
    TT_THROW(
        "Invalid pattern type: '{}'. Valid types are: all_to_all_device_unicast, all_hosts_random_sockets, "
        "all_device_broadcast",
        pattern_string);
}

std::vector<std::vector<EthCoord>> MeshSocketYamlParser::parse_eth_coord_mapping(const YAML::Node& yaml_node) {
    std::vector<std::vector<EthCoord>> array;

    TT_FATAL(yaml_node.IsSequence(), "Expected a sequence for 2D array");

    for (const auto& row : yaml_node) {
        TT_FATAL(row.IsSequence(), "Expected each row to be a sequence");
        std::vector<EthCoord> row_vector;
        row_vector.reserve(row.size());
        for (const auto& entry : row) {
            TT_FATAL(entry.size() == 5, "Expected ethernet core coordinates to be a sequence of 5 elements");
            row_vector.push_back(EthCoord{
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

void MeshSocketYamlParser::validate_memory_config(const MemoryConfig& memory) {
    TT_FATAL(!memory.fifo_size.empty(), "fifo_size must have at least one value");
    TT_FATAL(!memory.page_size.empty(), "page_size must have at least one value");
    TT_FATAL(!memory.data_size.empty(), "data_size must have at least one value");
    TT_FATAL(memory.num_transactions > 0, "num_transactions must be greater than 0");

    for (uint32_t fifo_size : memory.fifo_size) {
        TT_FATAL(fifo_size > 0, "All fifo_size values must be greater than 0");
    }
    for (uint32_t page_size : memory.page_size) {
        TT_FATAL(page_size > 0, "All page_size values must be greater than 0");
    }
    for (uint32_t data_size : memory.data_size) {
        TT_FATAL(data_size > 0, "All data_size values must be greater than 0");
    }
}

void MeshSocketYamlParser::validate_socket_config(
    const TestSocketConfig& socket_config, const MeshSocketTestContext& test_context) {
    const auto& distributed_context = test_context.get_distributed_context();
    const auto& mesh_graph = test_context.get_mesh_graph();
    const auto& rank_to_mesh_mapping = test_context.get_rank_to_mesh_mapping();
    const auto& mesh_device = test_context.get_mesh_device();

    auto world_size = *distributed_context->size();

    // 1. Check if sender and receiver ranks are < distributed context world size
    TT_FATAL(
        *socket_config.sender_rank < world_size,
        "Sender rank {} is >= world size {}",
        *socket_config.sender_rank,
        world_size);
    TT_FATAL(
        *socket_config.receiver_rank < world_size,
        "Receiver rank {} is >= world size {}",
        *socket_config.receiver_rank,
        world_size);

    TT_FATAL(socket_config.sender_rank != socket_config.receiver_rank, "Sender and receiver ranks must be different");

    auto sender_mesh_id = rank_to_mesh_mapping.at(socket_config.sender_rank);
    auto receiver_mesh_id = rank_to_mesh_mapping.at(socket_config.receiver_rank);

    // Get mesh shapes for validation
    auto sender_mesh_shape = mesh_graph.get_mesh_shape(sender_mesh_id);
    auto receiver_mesh_shape = mesh_graph.get_mesh_shape(receiver_mesh_id);

    // Create coordinate ranges for bounds checking
    tt::tt_metal::distributed::MeshCoordinateRange sender_coord_range(sender_mesh_shape);
    tt::tt_metal::distributed::MeshCoordinateRange receiver_coord_range(receiver_mesh_shape);

    // Collections to check for duplicate endpoints
    std::set<MeshCoordinate> sender_endpoints;
    std::set<MeshCoordinate> receiver_endpoints;
    auto device_compute_grid = mesh_device->compute_with_storage_grid_size();
    // 2. & 3. For each connection, validate mesh coordinates and check for duplicates
    for (const auto& connection : socket_config.connections) {
        const auto& sender_mesh_coord = connection.sender.mesh_coord;
        const auto& receiver_mesh_coord = connection.receiver.mesh_coord;

        // Validate sender mesh coordinate is within bounds
        TT_FATAL(
            sender_coord_range.contains(sender_mesh_coord),
            "Sender mesh coordinate {} is out of bounds for mesh shape {} (mesh_id {})",
            sender_mesh_coord,
            sender_mesh_shape,
            *sender_mesh_id);

        // Validate receiver mesh coordinate is within bounds
        TT_FATAL(
            receiver_coord_range.contains(receiver_mesh_coord),
            "Receiver mesh coordinate {} is out of bounds for mesh shape {} (mesh_id {})",
            receiver_mesh_coord,
            receiver_mesh_shape,
            *receiver_mesh_id);

        // Check for duplicate sender endpoints
        sender_endpoints.insert(sender_mesh_coord);

        // Check for duplicate receiver endpoints
        auto receiver_insert_result = receiver_endpoints.insert(receiver_mesh_coord);
        TT_FATAL(
            receiver_insert_result.second,
            "Duplicate receiver endpoint found at mesh coordinate {}",
            receiver_mesh_coord);

        // Validate core coordinates for each connection
        // Only checks if same rank
        if (distributed_context->rank() == socket_config.sender_rank) {
            TT_FATAL(
                connection.sender.core_coord.x < device_compute_grid.x &&
                    connection.sender.core_coord.y < device_compute_grid.y,
                "Sender core coordinate ({}, {}) is out of bounds for device compute grid ({}, {}) at mesh coordinate "
                "{}",
                connection.sender.core_coord.x,
                connection.sender.core_coord.y,
                device_compute_grid.x,
                device_compute_grid.y,
                sender_mesh_coord);
        }

        if (distributed_context->rank() == socket_config.receiver_rank) {
            TT_FATAL(
                connection.receiver.core_coord.x < device_compute_grid.x &&
                    connection.receiver.core_coord.y < device_compute_grid.y,
                "Receiver core coordinate ({}, {}) is out of bounds for device compute grid ({}, {}) at mesh "
                "coordinate {}",
                connection.receiver.core_coord.x,
                connection.receiver.core_coord.y,
                device_compute_grid.x,
                device_compute_grid.y,
                receiver_mesh_coord);
        }
    }
}

void MeshSocketYamlParser::print_test_configuration(const MeshSocketTestConfiguration& config) {
    log_info(tt::LogTest, "=== Parsed MeshSocket Test Configuration ===");

    // Print fabric configuration
    log_info(tt::LogTest, "Fabric Configuration:");
    log_info(tt::LogTest, "  Topology: {}", static_cast<int>(config.fabric_config.topology));

    // Print physical mesh configuration if present
    if (config.physical_mesh_config.has_value()) {
        const auto& physical_mesh = config.physical_mesh_config.value();
        log_info(tt::LogTest, "Physical Mesh Configuration:");
        if (!physical_mesh.mesh_descriptor_path.empty()) {
            log_info(tt::LogTest, "  Mesh Descriptor Path: {}", physical_mesh.mesh_descriptor_path);
        }
        if (!physical_mesh.eth_coord_mapping.empty()) {
            log_info(
                tt::LogTest,
                "  Ethernet Coordinate Mapping: {} rows with {} columns each",
                physical_mesh.eth_coord_mapping.size(),
                physical_mesh.eth_coord_mapping.empty() ? 0 : physical_mesh.eth_coord_mapping[0].size());
        }
    }

    // Print test configurations with detailed information
    log_info(tt::LogTest, "Parsed Test Configurations: {} tests", config.tests.size());
    for (size_t i = 0; i < config.tests.size(); ++i) {
        const auto& test = config.tests[i];
        log_info(tt::LogTest, "");
        log_info(tt::LogTest, "Test {}: '{}'", i + 1, test.name);

        if (test.num_iterations.has_value()) {
            log_info(tt::LogTest, "  Iterations: {}", test.num_iterations.value());
        }

        // Print memory config arrays
        log_info(tt::LogTest, "  Memory Config:");
        log_info(tt::LogTest, "    fifo_size: {} values", test.memory_config.fifo_size.size());
        log_info(tt::LogTest, "    page_size: {} values", test.memory_config.page_size.size());
        log_info(tt::LogTest, "    data_size: {} values", test.memory_config.data_size.size());
        log_info(tt::LogTest, "    num_transactions: {}", test.memory_config.num_transactions);

        // Print sockets if present
        if (test.sockets.has_value()) {
            log_info(tt::LogTest, "  Sockets: {} defined", test.sockets.value().size());
            for (size_t socket_idx = 0; socket_idx < test.sockets.value().size(); ++socket_idx) {
                const auto& socket = test.sockets.value()[socket_idx];
                log_info(tt::LogTest, "    Socket {}: {} connections", socket_idx + 1, socket.connections.size());
                log_info(
                    tt::LogTest,
                    "      Sender Rank: {}, Receiver Rank: {}",
                    *socket.sender_rank,
                    *socket.receiver_rank);

                // Print connection details
                for (size_t conn_idx = 0; conn_idx < socket.connections.size(); ++conn_idx) {
                    const auto& connection = socket.connections[conn_idx];
                    log_info(
                        tt::LogTest,
                        "      Connection {}: [{}, {}] -> [{}, {}]",
                        conn_idx + 1,
                        connection.sender.mesh_coord[0],
                        connection.sender.mesh_coord[1],
                        connection.receiver.mesh_coord[0],
                        connection.receiver.mesh_coord[1]);
                }
            }
        }

        // Print pattern expansions if present
        if (test.pattern_expansions.has_value()) {
            log_info(tt::LogTest, "  Pattern Expansions: {} defined", test.pattern_expansions.value().size());
            for (size_t pattern_idx = 0; pattern_idx < test.pattern_expansions.value().size(); ++pattern_idx) {
                const auto& pattern = test.pattern_expansions.value()[pattern_idx];
                std::string pattern_type;

                // switch if we add more patterns
                switch (pattern.type) {
                    case PatternType::AllToAllDevices: pattern_type = "all_to_all_device_unicast"; break;
                    case PatternType::AllHostsRandomSockets: pattern_type = "all_hosts_random_sockets"; break;
                    case PatternType::AllDeviceBroadcast: pattern_type = "all_device_broadcast"; break;
                    default: TT_THROW("Invalid pattern type: {}", static_cast<int>(pattern.type));
                }
                if (pattern.num_sockets.has_value()) {
                    log_info(
                        tt::LogTest,
                        "    Pattern {}: type={}, core_coord=({}, {}), num_sockets={}",
                        pattern_idx + 1,
                        pattern_type,
                        pattern.core_coord.x,
                        pattern.core_coord.y,
                        pattern.num_sockets.value());
                } else {
                    log_info(
                        tt::LogTest,
                        "    Pattern {}: type={}, core_coord=({}, {})",
                        pattern_idx + 1,
                        pattern_type,
                        pattern.core_coord.x,
                        pattern.core_coord.y);
                }
            }
        }
    }

    log_info(tt::LogTest, "=== End Parsed Configuration ===");
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
