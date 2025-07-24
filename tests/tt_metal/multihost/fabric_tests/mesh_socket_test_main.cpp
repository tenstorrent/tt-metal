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
    }
    log_info(tt::LogTest, "=== End Configuration Summary ===");
}

int main(int argc, char* argv[]) {
    std::string yaml_file_path;

    // Parse command line arguments
    if (argc < 2) {
        // Use default file if no argument provided
        yaml_file_path = "tests/tt_metal/multihost/fabric_tests/mesh_socket_test_config_example.yaml";
        log_info(tt::LogTest, "No YAML file specified, using default: {}", yaml_file_path);
    } else {
        yaml_file_path = argv[1];
        log_info(tt::LogTest, "Using YAML file: {}", yaml_file_path);
    }

    try {
        // Create parser and parse the configuration
        MeshSocketYamlParser parser;
        MeshSocketTestConfiguration config = parser.parse_file(yaml_file_path);

        log_info(tt::LogTest, "Successfully parsed YAML configuration from: {}", yaml_file_path);

        // Print summary of loaded configuration
        print_configuration_summary(config);

        // Basic validation
        log_info(tt::LogTest, "Running configuration validation...");
        parser.validate_configuration(config);
        log_info(tt::LogTest, "Configuration validation passed!");

        // Example of accessing specific test data
        if (!config.tests.empty()) {
            const auto& first_test = config.tests[0];
            log_info(
                tt::LogTest, "Example: First test '{}' has {} sockets", first_test.name, first_test.sockets.size());

            if (!first_test.sockets.empty()) {
                const auto& first_socket = first_test.sockets[0];
                log_info(tt::LogTest, "Example: First socket has {} connections", first_socket.connections.size());

                if (!first_socket.connections.empty()) {
                    const auto& first_connection = first_socket.connections[0];
                    log_info(
                        tt::LogTest,
                        "Example: First connection: [{}, {}] -> [{}, {}]",
                        first_connection.sender.mesh_coord[0],
                        first_connection.sender.mesh_coord[1],
                        first_connection.receiver.mesh_coord[0],
                        first_connection.receiver.mesh_coord[1]);
                }
            }
        }

        log_info(tt::LogTest, "YAML parsing and validation completed successfully!");
        return 0;

    } catch (const std::exception& e) {
        log_error(tt::LogTest, "Error processing YAML file: {}", e.what());
        return 1;
    }
}
