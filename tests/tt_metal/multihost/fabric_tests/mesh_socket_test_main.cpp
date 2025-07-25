// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <string>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include "tests/tt_metal/multihost/fabric_tests/mesh_socket_yaml_parser.hpp"

using namespace tt::tt_fabric::mesh_socket_tests;

void print_configuration_summary(const MeshSocketTestConfiguration& config) {
    log_info(tt::LogTest, "=== MeshSocket Test Configuration Summary ===");

    // Print fabric configuration
    log_info(tt::LogTest, "Fabric Configuration:");
    log_info(tt::LogTest, "  Topology: {}", static_cast<int>(config.fabric_config.topology));
    if (config.fabric_config.routing_type.has_value()) {
        log_info(tt::LogTest, "  Routing Type: {}", static_cast<int>(config.fabric_config.routing_type.value()));
    }

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

    // Print test configurations
    log_info(tt::LogTest, "Test Configurations: {} tests defined", config.tests.size());
    for (size_t i = 0; i < config.tests.size(); ++i) {
        const auto& test = config.tests[i];
        log_info(tt::LogTest, "  Test {}: '{}'", i + 1, test.name);
        if (test.num_iterations.has_value()) {
            log_info(tt::LogTest, "    Iterations: {}", test.num_iterations.value());
        }
        log_info(
            tt::LogTest,
            "    Memory Config: fifo_size={}, page_size={}, data_size={}",
            test.memory_config.fifo_size,
            test.memory_config.page_size,
            test.memory_config.data_size);
        log_info(tt::LogTest, "    Sockets: {} defined", test.sockets.size());

        // Print detailed socket information
        for (size_t socket_idx = 0; socket_idx < test.sockets.size(); ++socket_idx) {
            const auto& socket = test.sockets[socket_idx];
            log_info(tt::LogTest, "      Socket {}: {} connections", socket_idx + 1, socket.connections.size());
            log_info(
                tt::LogTest, "        Sender Rank: {}, Receiver Rank: {}", *socket.sender_rank, *socket.receiver_rank);
            // Print connection details
            for (size_t conn_idx = 0; conn_idx < socket.connections.size(); ++conn_idx) {
                const auto& connection = socket.connections[conn_idx];
                log_info(
                    tt::LogTest,
                    "        Connection {}: [{}, {}] -> [{}, {}]",
                    conn_idx + 1,
                    connection.sender.mesh_coord[0],
                    connection.sender.mesh_coord[1],
                    connection.receiver.mesh_coord[0],
                    connection.receiver.mesh_coord[1]);
            }
        }
    }
    log_info(tt::LogTest, "=== End Configuration Summary ===");
}

int main(int argc, char* argv[]) {
    std::vector<std::string> input_args(argv, argv + argc);
    // Parse command line and YAML configurations
    CmdlineParser cmdline_parser(input_args);

    if (cmdline_parser.has_help_option()) {
        cmdline_parser.print_help();
        return 0;
    }

    auto yaml_file_path = cmdline_parser.get_yaml_config_path();
    if (!yaml_file_path.has_value()) {
        log_error(tt::LogTest, "No YAML file specified. Use --help for usage information.");
        return 1;
    }

    try {
        // Create parser and parse the configuration
        MeshSocketYamlParser parser;
        MeshSocketTestConfiguration config = parser.parse_file(yaml_file_path.value());

        log_info(tt::LogTest, "Successfully parsed YAML configuration from: {}", yaml_file_path.value());

        // Print summary of loaded configuration
        print_configuration_summary(config);

        // Basic validation
        log_info(tt::LogTest, "Running configuration validation...");
        parser.validate_configuration(config);
        log_info(tt::LogTest, "Configuration validation passed!");

        // Example of accessing all test data
        log_info(tt::LogTest, "=== Detailed Test Data Access Example ===");
        for (size_t test_idx = 0; test_idx < config.tests.size(); ++test_idx) {
            const auto& test = config.tests[test_idx];
            log_info(tt::LogTest, "Test {}: '{}' has {} sockets", test_idx + 1, test.name, test.sockets.size());

            for (size_t socket_idx = 0; socket_idx < test.sockets.size(); ++socket_idx) {
                const auto& socket = test.sockets[socket_idx];
                log_info(
                    tt::LogTest,
                    "  Socket {}: {} connections, Sender Rank: {}, Receiver Rank: {}",
                    socket_idx + 1,
                    socket.connections.size(),
                    *socket.sender_rank,
                    *socket.receiver_rank);

                for (size_t conn_idx = 0; conn_idx < socket.connections.size(); ++conn_idx) {
                    const auto& connection = socket.connections[conn_idx];
                    log_info(
                        tt::LogTest,
                        "    Connection {}: [{}, {}] -> [{}, {}]",
                        conn_idx + 1,
                        connection.sender.mesh_coord[0],
                        connection.sender.mesh_coord[1],
                        connection.receiver.mesh_coord[0],
                        connection.receiver.mesh_coord[1]);
                }
            }
        }
        log_info(tt::LogTest, "=== End Detailed Test Data ===");

        log_info(tt::LogTest, "YAML parsing and validation completed successfully!");
        return 0;

    } catch (const std::exception& e) {
        log_error(tt::LogTest, "Error processing YAML file: {}", e.what());
        return 1;
    }
}
