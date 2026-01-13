// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
#include "tt_fabric_test_constants.hpp"

using tt::tt_fabric::fabric_tests::DEFAULT_BUILT_TESTS_DUMP_FILE;
using tt::tt_fabric::fabric_tests::OUTPUT_DIR;

const std::unordered_map<Topology, FabricConfig> TestFixture::topology_to_fabric_config_map = {
    {Topology::NeighborExchange, FabricConfig::FABRIC_1D_NEIGHBOR_EXCHANGE},
    {Topology::Linear, FabricConfig::FABRIC_1D},
    {Topology::Ring, FabricConfig::FABRIC_1D_RING},
    {Topology::Mesh, FabricConfig::FABRIC_2D},
};

const std::unordered_map<std::pair<Topology, std::string>, FabricConfig, tt::tt_fabric::fabric_tests::pair_hash>
    TestFixture::torus_topology_to_fabric_config_map = {
        {{Topology::Torus, "X"}, FabricConfig::FABRIC_2D_TORUS_X},
        {{Topology::Torus, "Y"}, FabricConfig::FABRIC_2D_TORUS_Y},
        {{Topology::Torus, "XY"}, FabricConfig::FABRIC_2D_TORUS_XY},
};

int main(int argc, char** argv) {
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
    bool use_dynamic_policies = true;  // Default to dynamic

    if (auto yaml_path = cmdline_parser.get_yaml_config_path()) {
        YamlConfigParser yaml_parser;
        auto parsed_yaml = yaml_parser.parse_file(yaml_path.value());
        raw_test_configs = std::move(parsed_yaml.test_configs);

        // Check if YAML explicitly provided allocation_policies
        if (parsed_yaml.allocation_policies.has_value()) {
            allocation_policies = parsed_yaml.allocation_policies.value();
            use_dynamic_policies = false;  // User provided explicit policies
        }

        if (parsed_yaml.physical_mesh_config.has_value()) {
            physical_mesh_config = parsed_yaml.physical_mesh_config;
        }
    } else {
        log_error(
            tt::LogTest,
            "No YAML config file path specified. Please use --test_config <file_path> to specify the test config. Use "
            "--help for more information.");
        return 1;
    }

    log_info(tt::LogTest, "Starting Test");

    fixture->init(physical_mesh_config);

    TestContext test_context;
    test_context.init(fixture, allocation_policies, use_dynamic_policies);

    // Configure progress monitoring from cmdline flags
    if (cmdline_parser.show_progress()) {
        ProgressMonitorConfig progress_config;
        progress_config.enabled = true;
        progress_config.poll_interval_seconds = cmdline_parser.get_progress_interval();
        progress_config.hung_threshold_seconds = cmdline_parser.get_hung_threshold();

        test_context.enable_progress_monitoring(progress_config);
    }

    bool has_bandwidth_tests = std::any_of(raw_test_configs.begin(), raw_test_configs.end(), [](const auto& config) {
        return config.performance_test_mode == PerformanceTestMode::BANDWIDTH;
    });

    // Initialize CSV file for bandwidth results if any of the configs have bandwidth test mode set
    if (has_bandwidth_tests) {
        test_context.initialize_bandwidth_results_csv_file();
    }

    bool has_latency_tests = std::any_of(raw_test_configs.begin(), raw_test_configs.end(), [](const auto& config) {
        return config.performance_test_mode == PerformanceTestMode::LATENCY;
    });

    // Initialize CSV file for latency results if any of the configs have latency test mode set
    if (has_latency_tests) {
        test_context.initialize_latency_results_csv_file();
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
            std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
            std::string(OUTPUT_DIR);
        if (!std::filesystem::exists(dump_file_dir)) {
            std::filesystem::create_directory(dump_file_dir);
        }

        std::string dump_file = cmdline_parser.get_built_tests_dump_file_name(DEFAULT_BUILT_TESTS_DUMP_FILE);
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
    uint32_t tests_ran = 0;
    for (auto& test_config : raw_test_configs) {
        if (!cmdline_parser.check_filter(test_config, true)) {
            log_info(tt::LogTest, "Skipping Test Group: {} due to filter policy", test_config.name);
            continue;
        }
        if (builder.should_skip_test_on_platform(test_config)) {
            log_info(tt::LogTest, "Skipping Test Group: {} due to platform skip policy", test_config.name);
            continue;
        }
        log_info(tt::LogTest, "Running Test Group: {}", test_config.name);

        const auto& topology = test_config.fabric_setup.topology;
        const auto& fabric_tensix_config = test_config.fabric_setup.fabric_tensix_config.value();
        if (test_config.performance_test_mode != PerformanceTestMode::NONE) {
            tt::tt_metal::MetalContext::instance().rtoptions().set_enable_fabric_bw_telemetry(true);
        }

        log_info(
            tt::LogTest,
            "Opening devices with topology: {} and fabric_tensix_config: {}",
            topology,
            fabric_tensix_config);

        bool open_devices_success = test_context.open_devices(test_config.fabric_setup);
        if (!open_devices_success) {
            log_warning(
                tt::LogTest, "Skipping Test Group: {} due to unsupported fabric configuration", test_config.name);
            continue;
        }

        // Validate device frequencies for performance tests. Validation runs only once
        // since device frequencies are cached in TestFixture for its lifetime.
        if (test_config.performance_test_mode != PerformanceTestMode::NONE) {
            if (!fixture->validate_device_frequencies_for_performance_tests()) {
                test_context.close_devices();
                return 1;  // Hard exit - cannot run performance benchmarks with invalid frequencies
            }
        }

        // Check topology-based skip conditions after devices are opened
        if (builder.should_skip_test_on_topology(test_config)) {
            log_info(tt::LogTest, "Skipping Test Group: {} due to topology skip policy", test_config.name);
            test_context.close_devices();
            continue;
        }
        tests_ran++;
        device_opened = true;

        for (uint32_t iter = 0; iter < test_config.num_top_level_iterations; ++iter) {
            log_info(tt::LogTest, "Starting top-level iteration {}/{}", iter + 1, test_config.num_top_level_iterations);

            log_info(tt::LogTest, "Building tests");
            auto built_tests = builder.build_tests({test_config}, cmdline_parser);

            // Set performance test mode and line sync for this test group
            test_context.set_performance_test_mode(test_config.performance_test_mode);
            // Enable telemetry for both benchmark and latency modes to ensure buffer clearing
            test_context.set_telemetry_enabled(test_config.performance_test_mode != PerformanceTestMode::NONE);
            // Set skip_packet_validation flag
            test_context.set_skip_packet_validation(test_config.skip_packet_validation);

            // Set code profiling enabled based on rtoptions
            auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
            test_context.set_code_profiling_enabled(rtoptions.get_enable_fabric_code_profiling_rx_ch_fwd());

            for (auto& built_test : built_tests) {
                log_info(tt::LogTest, "Running Test: {}", built_test.parametrized_name);

                // Prepare allocator and memory maps for this specific test
                test_context.prepare_for_test(built_test);

                test_context.setup_devices();
                log_info(tt::LogTest, "Device setup complete");

                test_context.process_traffic_config(built_test);
                log_info(tt::LogTest, "Traffic config processed");

                // Setup latency test mode AFTER process_traffic_config so that senders_/receivers_ maps are populated
                if (built_test.performance_test_mode == PerformanceTestMode::LATENCY) {
                    test_context.setup_latency_test_mode(built_test);
                }

                // Clear code profiling buffers before test execution
                if (test_context.get_code_profiling_enabled()) {
                    test_context.clear_code_profiling_buffers();
                }

                if (dump_built_tests) {
                    YamlTestConfigSerializer::dump({built_test}, output_stream);
                }

                log_info(tt::LogTest, "Compiling programs");
                test_context.compile_programs();

                // multi-host barrier to synchronize before starting the test (as we could be clearing out addresses)
                fixture->barrier();

                log_info(tt::LogTest, "Launching programs");
                test_context.launch_programs();

                log_info(tt::LogTest, "Waiting for programs");
                test_context.wait_for_programs_with_progress();
                log_info(tt::LogTest, "Test {} Finished.", built_test.parametrized_name);

                test_context.process_telemetry_data(built_test);

                // Read and report code profiling results
                if (test_context.get_code_profiling_enabled()) {
                    test_context.read_code_profiling_results();
                    test_context.report_code_profiling_results();
                }

                test_context.validate_results();

                // Performance profiling (bandwidth mode)
                if (test_context.get_performance_test_mode() == PerformanceTestMode::BANDWIDTH) {
                    test_context.profile_results(built_test);
                }

                // Latency measurement (latency test mode)
                if (test_context.get_performance_test_mode() == PerformanceTestMode::LATENCY) {
                    test_context.collect_latency_results();
                    test_context.report_latency_results(built_test);
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

    tt::tt_metal::MetalContext::instance().rtoptions().set_enable_fabric_bw_telemetry(false);

    // Generate summaries after all tests have run
    if (has_bandwidth_tests) {
        test_context.generate_bandwidth_summary();
    }
    if (has_latency_tests) {
        test_context.generate_latency_summary();
    }

    // Setup CSV files for CI to upload (handles both bandwidth and latency)
    if (has_bandwidth_tests || has_latency_tests) {
        test_context.setup_ci_artifacts();
    }

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

    auto total_tests_count = raw_test_configs.size();
    if (tests_ran < total_tests_count) {
        log_warning(
            tt::LogTest,
            "{} out of {} test groups did not run (filtered, skipped, or unsupported)",
            total_tests_count - tests_ran,
            total_tests_count);
    }

    if (device_opened) {
        log_info(tt::LogTest, "All tests completed successfully");
    } else {
        log_info(tt::LogTest, "No tests found for provided filter");
    }
    return 0;
}
