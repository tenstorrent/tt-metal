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

#include "tt_fabric_test_context.hpp"

const std::unordered_map<std::pair<Topology, RoutingType>, FabricConfig, tt::tt_fabric::fabric_tests::pair_hash>
    TestFixture::topology_to_fabric_config_map = {
        {{Topology::Linear, RoutingType::LowLatency}, FabricConfig::FABRIC_1D},
        {{Topology::Ring, RoutingType::LowLatency}, FabricConfig::FABRIC_1D_RING},
        {{Topology::Mesh, RoutingType::LowLatency}, FabricConfig::FABRIC_2D},
        {{Topology::Mesh, RoutingType::Dynamic}, FabricConfig::FABRIC_2D_DYNAMIC},
};

const std::
    unordered_map<std::tuple<Topology, std::string, RoutingType>, FabricConfig, tt::tt_fabric::fabric_tests::tuple_hash>
        TestFixture::torus_topology_to_fabric_config_map = {
            {{Topology::Torus, "X", RoutingType::LowLatency}, FabricConfig::FABRIC_2D_TORUS_X},
            {{Topology::Torus, "X", RoutingType::Dynamic}, FabricConfig::FABRIC_2D_DYNAMIC_TORUS_X},
            {{Topology::Torus, "Y", RoutingType::LowLatency}, FabricConfig::FABRIC_2D_TORUS_Y},
            {{Topology::Torus, "Y", RoutingType::Dynamic}, FabricConfig::FABRIC_2D_DYNAMIC_TORUS_Y},
            {{Topology::Torus, "XY", RoutingType::LowLatency}, FabricConfig::FABRIC_2D_TORUS_XY},
            {{Topology::Torus, "XY", RoutingType::Dynamic}, FabricConfig::FABRIC_2D_DYNAMIC_TORUS_XY},
};

int main(int argc, char** argv) {
    log_info(tt::LogTest, "Starting Test");
    std::vector<std::string> input_args(argv, argv + argc);

    auto fixture = std::make_shared<TestFixture>();

    // Parse command line and YAML configurations
    CmdlineParser cmdline_parser(input_args);

    if (cmdline_parser.has_help_option()) {
        cmdline_parser.print_help();
        return 0;
    }
    std::vector<ParsedTestConfig> raw_test_configs;
    tt::tt_fabric::fabric_tests::AllocatorPolicies allocation_policies;
    std::optional<tt::tt_fabric::fabric_tests::PhysicalMeshConfig> physical_mesh_config = std::nullopt;
    if (auto yaml_path = cmdline_parser.get_yaml_config_path()) {
        YamlConfigParser yaml_parser;
        auto parsed_yaml = yaml_parser.parse_file(yaml_path.value());
        raw_test_configs = std::move(parsed_yaml.test_configs);
        if (parsed_yaml.allocation_policies.has_value()) {
            allocation_policies = parsed_yaml.allocation_policies.value();
        }
        if (parsed_yaml.physical_mesh_config.has_value()) {
            physical_mesh_config = parsed_yaml.physical_mesh_config;
        }
    } else {
        raw_test_configs = cmdline_parser.generate_default_configs();
    }

    fixture->init(physical_mesh_config);

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
        master_seed = test_context.get_randomized_master_seed();
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

        // dump physical mesh first
        if (physical_mesh_config.has_value()) {
            YamlTestConfigSerializer::dump(physical_mesh_config.value(), output_stream);
        }

        // dump allocation policies second
        YamlTestConfigSerializer::dump(allocation_policies, output_stream);
    }

    bool device_opened = false;
    for (auto& test_config : raw_test_configs) {
        if (!cmdline_parser.check_filter(test_config, true)) {
            log_info(tt::LogTest, "Skipping Test Group: {} due to filter policy", test_config.name);
            continue;
        }
        log_info(tt::LogTest, "Running Test Group: {}", test_config.name);

        const auto& topology = test_config.fabric_setup.topology;
        const auto& routing_type = test_config.fabric_setup.routing_type.value();
        const auto& fabric_tensix_config = test_config.fabric_setup.fabric_tensix_config.value();
        if (test_config.benchmark_mode) {
            tt::tt_metal::MetalContext::instance().rtoptions().set_enable_fabric_telemetry(true);
        }

        log_info(
            tt::LogTest,
            "Opening devices with topology: {}, routing type: {}, and fabric_tensix_config: {}",
            topology,
            routing_type,
            fabric_tensix_config);
        test_context.open_devices(test_config.fabric_setup);
        device_opened = true;

        for (uint32_t iter = 0; iter < test_config.num_top_level_iterations; ++iter) {
            log_info(tt::LogTest, "Starting top-level iteration {}/{}", iter + 1, test_config.num_top_level_iterations);

            log_info(tt::LogTest, "Building tests");
            auto built_tests = builder.build_tests({test_config}, cmdline_parser);

            // Set benchmark mode and line sync for this test group
            test_context.set_benchmark_mode(test_config.benchmark_mode);
            test_context.set_telemetry_enabled(test_config.benchmark_mode);

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

                test_context.wait_for_programs();
                log_info(tt::LogTest, "Test {} Finished.", built_test.name);

                test_context.process_telemetry_data(built_test);

                test_context.validate_results();
                log_info(tt::LogTest, "Test {} Results validated.", built_test.name);

                if (test_context.get_benchmark_mode()) {
                    test_context.profile_results(built_test);
                }
                if (test_context.get_telemetry_enabled()) {
                    test_context.clear_telemetry();
                }
                // Synchronize across all hosts after running the current test variant
                fixture->barrier();
                test_context.reset_devices();
            }
        }
    }

    test_context.close_devices();

    tt::tt_metal::MetalContext::instance().rtoptions().set_enable_fabric_telemetry(false);

    // Check if any tests failed validation and throw at the end
    if (test_context.has_test_failures()) {
        const auto& failed_tests = test_context.get_all_failed_tests();
        log_error(tt::LogTest, "=== FINAL TEST SUMMARY ===");
        log_error(tt::LogTest, "Total failed tests: {}", failed_tests.size());
        log_error(tt::LogTest, "Failed tests:");
        for (const auto& failed_test : failed_tests) {
            log_error(tt::LogTest, "  - {}", failed_test);
        }
        TT_THROW("Some tests failed golden comparison validation. See summary above.");
    }

    if (device_opened) {
        log_info(tt::LogTest, "All tests completed successfully");
    } else {
        log_info(tt::LogTest, "No tests found for provided filter");
    }
    return 0;
}
