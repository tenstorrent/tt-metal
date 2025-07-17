// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <random>
#include <filesystem>
#include <optional>
#include <iomanip>
#include <sstream>
#include <memory>
#include <enchantum/enchantum.hpp>

#include "tt_fabric_test_context.hpp"

const std::unordered_map<std::pair<Topology, RoutingType>, FabricConfig, tt::tt_fabric::fabric_tests::pair_hash>
    TestFixture::topology_to_fabric_config_map = {
        {{Topology::Linear, RoutingType::LowLatency}, FabricConfig::FABRIC_1D},
        {{Topology::Ring, RoutingType::LowLatency}, FabricConfig::FABRIC_1D_RING},
        {{Topology::Mesh, RoutingType::LowLatency}, FabricConfig::FABRIC_2D},
        {{Topology::Mesh, RoutingType::Dynamic}, FabricConfig::FABRIC_2D_DYNAMIC},
};

int main(int argc, char** argv) {
    std::vector<std::string> input_args(argv, argv + argc);

    auto fixture = std::make_shared<TestFixture>();
    fixture->init();

    // Parse command line and YAML configurations
    CmdlineParser cmdline_parser(input_args);

    if (cmdline_parser.has_help_option()) {
        cmdline_parser.print_help();
        return 0;
    }

    std::vector<ParsedTestConfig> raw_test_configs;
    tt::tt_fabric::fabric_tests::AllocatorPolicies allocation_policies;

    if (auto yaml_path = cmdline_parser.get_yaml_config_path()) {
        YamlConfigParser yaml_parser;
        auto parsed_yaml = yaml_parser.parse_file(yaml_path.value());
        raw_test_configs = std::move(parsed_yaml.test_configs);
        if (parsed_yaml.allocation_policies.has_value()) {
            allocation_policies = parsed_yaml.allocation_policies.value();
        }
    } else {
        raw_test_configs = cmdline_parser.generate_default_configs();
    }

    TestContext test_context;
    test_context.init(fixture, allocation_policies);

    // Initialize CSV file for bandwidth results if any of the configs have benchmark mode set
    for (const auto& config : raw_test_configs) {
        if (config.benchmark_mode) {
            test_context.initialize_csv_file();
            break;
        }
    }

    cmdline_parser.apply_overrides(raw_test_configs);

    if (raw_test_configs.empty()) {
        log_fatal(tt::LogTest, "No test configurations loaded or generated. Exiting.");
        return 1;
    }

    std::optional<uint32_t> master_seed = cmdline_parser.get_master_seed();
    if (!master_seed.has_value()) {
        master_seed = std::random_device()();
        log_info(tt::LogTest, "No master seed provided. Using randomly generated seed: {}", master_seed.value());
    }
    std::mt19937 gen(master_seed.value());

    // fixture is passed twice since it implements both interfaces
    // the builder object does the initial processing of the tests parsed from yaml/cmd line and tries to fill
    // any gaps/optionals/missing values
    TestConfigBuilder builder(*fixture, *fixture, gen);

    std::ofstream output_stream;
    bool dump_built_tests = cmdline_parser.dump_built_tests();
    if (dump_built_tests) {
        std::filesystem::path dump_file_dir =
            std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / output_dir;
        if (!std::filesystem::exists(dump_file_dir)) {
            std::filesystem::create_directory(dump_file_dir);
        }

        std::string dump_file = cmdline_parser.get_built_tests_dump_file_name(default_built_tests_dump_file);
        std::filesystem::path dump_file_path = dump_file_dir / dump_file;
        output_stream.open(dump_file_path, std::ios::out | std::ios::trunc);

        // dump allocation policies first
        YamlTestConfigSerializer::dump(allocation_policies, output_stream);
    }

    for (auto& test_config : raw_test_configs) {
        log_info(tt::LogTest, "Running Test Group: {}", test_config.name);

        const auto& topology = test_config.fabric_setup.topology;
        const auto& routing_type = test_config.fabric_setup.routing_type.value();
        log_info(tt::LogTest, "Opening devices with topology: {} and routing type: {}", topology, routing_type);
        test_context.open_devices(topology, routing_type);

        log_info(tt::LogTest, "Building tests");
        auto built_tests = builder.build_tests({test_config});

        // Set benchmark mode and line sync for this test group
        test_context.set_benchmark_mode(test_config.benchmark_mode);

        for (auto& built_test : built_tests) {
            log_info(tt::LogTest, "Running Test: {}", built_test.name);

            test_context.setup_devices();
            log_info(tt::LogTest, "Device setup complete");

            test_context.process_traffic_config(built_test);
            log_info(tt::LogTest, "Traffic config processed");

            // Initialize sync memory if line sync is enabled
            test_context.initialize_sync_memory();

            if (dump_built_tests) {
                YamlTestConfigSerializer::dump({built_test}, output_stream);
            }

            log_info(tt::LogTest, "Compiling programs");
            test_context.compile_programs();

            log_info(tt::LogTest, "Launching programs");
            test_context.launch_programs();

            test_context.wait_for_prorgams();
            log_info(tt::LogTest, "Test {} Finished.", built_test.name);

            test_context.validate_results();
            log_info(tt::LogTest, "Test {} Results validated.", built_test.name);

            if (test_context.get_benchmark_mode()) {
                test_context.profile_results(built_test);
            }

            test_context.reset_devices();
        }
    }

    test_context.close_devices();

    return 0;
}
