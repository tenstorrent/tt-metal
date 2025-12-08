// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cxxopts.hpp>
#include <filesystem>
#include <iostream>

#include "generate_traffic_tests.hpp"

namespace fs = std::filesystem;
using namespace tt::scaleout_tools;

int main(int argc, char* argv[]) {
    cxxopts::Options options("generate_traffic_tests", "Generate traffic test YAML from cabling descriptors");

    // clang-format off
    options.add_options()
        ("c,cabling-descriptor-path", "Path to the cabling descriptor textproto file", cxxopts::value<std::string>())
        ("o,output-path", "Path to output traffic test YAML file", cxxopts::value<std::string>()->default_value("traffic_tests.yaml"))
        ("m,mgd-output-path", "Path to output MGD file (auto-generated)", cxxopts::value<std::string>())
        ("e,existing-mgd-path", "Path to existing MGD file (skip MGD generation)", cxxopts::value<std::string>())
        ("p,profile", "Test profile: sanity, stress, benchmark, coverage", cxxopts::value<std::string>()->default_value("sanity"))
        ("n,name-prefix", "Prefix for generated test names", cxxopts::value<std::string>()->default_value(""))
        ("flow-control", "Include flow control tests", cxxopts::value<bool>()->default_value("false"))
        ("no-sync", "Disable sync for tests", cxxopts::value<bool>()->default_value("false"))
        ("skip", "Platforms to skip (comma-separated, e.g., GALAXY,BLACKHOLE)", cxxopts::value<std::string>())
        ("v,verbose", "Enable verbose output", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print help message");
    // clang-format on

    try {
        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            std::cout << "\nProfiles:\n";
            std::cout << "  sanity    - Quick functional validation (default)\n";
            std::cout << "  stress    - High-volume stress testing with flow control\n";
            std::cout << "  benchmark - Performance measurement focused\n";
            std::cout << "  coverage  - Full coverage with all patterns\n";
            std::cout << "\nExamples:\n";
            std::cout << "  # Generate sanity tests for dual T3K\n";
            std::cout << "  ./generate_traffic_tests \\\n";
            std::cout << "      --cabling-descriptor-path tools/tests/scaleout/cabling_descriptors/dual_t3k.textproto \\\n";
            std::cout << "      --output-path tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_dual_t3k_auto.yaml \\\n";
            std::cout << "      --mgd-output-path tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_auto_mgd.textproto\n";
            std::cout << "\n  # Generate stress tests with existing MGD\n";
            std::cout << "  ./generate_traffic_tests \\\n";
            std::cout << "      --cabling-descriptor-path 16_n300_lb_cluster.textproto \\\n";
            std::cout << "      --existing-mgd-path existing_mgd.textproto \\\n";
            std::cout << "      --profile stress\n";
            return 0;
        }

        if (!result.count("cabling-descriptor-path")) {
            std::cerr << "Error: --cabling-descriptor-path is required\n";
            std::cerr << options.help() << std::endl;
            return 1;
        }

        const fs::path cabling_path = result["cabling-descriptor-path"].as<std::string>();
        const fs::path output_path = result["output-path"].as<std::string>();
        const bool verbose = result["verbose"].as<bool>();

        // Parse profile
        std::string profile_str = result["profile"].as<std::string>();
        TrafficTestConfig config;

        if (profile_str == "sanity") {
            config = get_sanity_config();
        } else if (profile_str == "stress") {
            config = get_stress_config();
        } else if (profile_str == "benchmark") {
            config = get_benchmark_config();
        } else if (profile_str == "coverage") {
            config = get_coverage_config();
        } else {
            std::cerr << "Error: Unknown profile '" << profile_str << "'. Use: sanity, stress, benchmark, coverage\n";
            return 1;
        }

        // Override config with command line options
        if (result.count("mgd-output-path")) {
            config.mgd_output_path = result["mgd-output-path"].as<std::string>();
            config.generate_mgd = true;
        } else {
            config.generate_mgd = false;
        }

        if (result.count("existing-mgd-path")) {
            config.existing_mgd_path = result["existing-mgd-path"].as<std::string>();
            config.generate_mgd = false;
        }

        if (result.count("name-prefix")) {
            config.test_name_prefix = result["name-prefix"].as<std::string>();
        }

        if (result["flow-control"].as<bool>()) {
            config.include_flow_control = true;
        }

        if (result["no-sync"].as<bool>()) {
            config.include_sync = false;
        }

        if (result.count("skip")) {
            std::string skip_str = result["skip"].as<std::string>();
            std::stringstream ss(skip_str);
            std::string platform;
            while (std::getline(ss, platform, ',')) {
                // Trim whitespace
                platform.erase(0, platform.find_first_not_of(" \t"));
                platform.erase(platform.find_last_not_of(" \t") + 1);
                if (!platform.empty()) {
                    config.skip_platforms.push_back(platform);
                }
            }
        }

        // Generate traffic tests
        generate_traffic_tests(cabling_path, output_path, config, verbose);

        std::cout << "Successfully generated traffic tests: " << output_path << "\n";
        if (config.generate_mgd && !config.mgd_output_path.empty()) {
            std::cout << "Also generated MGD: " << config.mgd_output_path << "\n";
        }

        return 0;

    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing options: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
