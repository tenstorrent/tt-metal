// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <string>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include "tests/tt_metal/multihost/fabric_tests/mesh_socket_yaml_parser.hpp"
#include "tests/tt_metal/multihost/fabric_tests/mesh_socket_test_context.hpp"

using namespace tt::tt_fabric::mesh_socket_tests;

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

    // Create parser and parse the configuration
    MeshSocketTestConfiguration config = MeshSocketYamlParser::parse_file(yaml_file_path.value());

    log_info(tt::LogTest, "Successfully parsed YAML configuration from: {}", yaml_file_path.value());

    if (cmdline_parser.print_configs()) {
        MeshSocketYamlParser::print_test_configuration(config);
        return 0;
    }

    // Create and run the test runner
    log_info(tt::LogTest, "Creating MeshSocketTestContext...");
    MeshSocketTestContext test_context(config);

    // Initialize the runner (sets up fabric and MeshDevice)
    test_context.initialize();

    // Run all tests defined in the configuration
    test_context.run_all_tests();

    // Cleanup is handled automatically by destructor
    return 0;
}
