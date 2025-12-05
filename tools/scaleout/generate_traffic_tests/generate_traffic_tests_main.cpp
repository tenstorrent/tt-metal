// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cxxopts.hpp>
#include <filesystem>
#include <iostream>
#include <sstream>

#include "generate_traffic_tests.hpp"

namespace fs = std::filesystem;
using namespace tt::scaleout_tools;

namespace {

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        // Trim whitespace
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    return result;
}

std::vector<uint32_t> parse_sizes(const std::string& s) {
    std::vector<uint32_t> result;
    for (const auto& part : split(s, ',')) {
        result.push_back(std::stoul(part));
    }
    return result;
}

}  // namespace

int main(int argc, char* argv[]) {
    cxxopts::Options options(
        "generate_traffic_tests",
        "Generate fabric traffic test YAML files from cabling descriptors.\n\n"
        "Examples:\n"
        "  # Quick sanity tests\n"
        "  ./generate_traffic_tests -c cluster.textproto -o tests.yaml -m mgd.textproto\n\n"
        "  # Stress testing with all categories\n"
        "  ./generate_traffic_tests -c cluster.textproto -o tests.yaml -p stress\n\n"
        "  # Custom: only all-to-all with specific sizes\n"
        "  ./generate_traffic_tests -c cluster.textproto -o tests.yaml \\\n"
        "      --disable-simple --disable-inter-mesh --sizes 1024,4096 --packets 500");

    // clang-format off
    options.add_options()
        ("c,cabling-descriptor-path", "Path to cabling descriptor textproto (required)",
            cxxopts::value<std::string>())
        ("o,output-path", "Output YAML file path",
            cxxopts::value<std::string>()->default_value("traffic_tests.yaml"))
        ("m,mgd-output-path", "Path to generate MGD file",
            cxxopts::value<std::string>())
        ("e,existing-mgd-path", "Use existing MGD file (skip generation)",
            cxxopts::value<std::string>())
        ("p,profile", "Test profile: sanity, stress, benchmark",
            cxxopts::value<std::string>()->default_value("sanity"))

        // Packet configuration
        ("sizes", "Packet sizes (comma-separated, e.g., 1024,2048,4096)",
            cxxopts::value<std::string>())
        ("packets", "Number of packets per sender",
            cxxopts::value<uint32_t>())
        ("noc-types", "NoC types (comma-separated, e.g., unicast_write,atomic_inc)",
            cxxopts::value<std::string>())

        // Test category toggles (disable specific tests)
        ("disable-simple", "Disable simple unicast test")
        ("disable-inter-mesh", "Disable inter-mesh test")
        ("disable-all-to-all", "Disable all-to-all test")
        ("enable-random", "Enable random pairing test")
        ("enable-all-to-one", "Enable all-to-one convergence test")
        ("enable-flow-control", "Enable flow control stress test")
        ("enable-sequential", "Enable sequential all-to-all test")

        // Other options
        ("n,name-prefix", "Prefix for test names",
            cxxopts::value<std::string>()->default_value(""))
        ("skip", "Platforms to skip (comma-separated)",
            cxxopts::value<std::string>())
        ("no-sync", "Disable sync between devices")
        ("v,verbose", "Verbose output")
        ("h,help", "Print help");
    // clang-format on

    try {
        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << "\n";
            std::cout << "Profiles:\n";
            std::cout << "  sanity    - Tests: simple, inter-mesh, all-to-all\n";
            std::cout << "              Packets: 100, Sizes: 1024,2048\n\n";
            std::cout << "  stress    - Tests: all categories\n";
            std::cout << "              Packets: 1000, Sizes: 1024,2048,4096\n\n";
            std::cout << "  benchmark - Tests: simple, inter-mesh, all-to-all, flow-control\n";
            std::cout << "              Packets: 1000, Sizes: 512,1024,2048,4096,8192\n";
            return 0;
        }

        if (!result.count("cabling-descriptor-path")) {
            std::cerr << "Error: --cabling-descriptor-path is required\n";
            std::cerr << "Use --help for usage\n";
            return 1;
        }

        // Parse profile
        std::string profile_str = result["profile"].as<std::string>();
        TrafficTestConfig config;

        if (profile_str == "sanity") {
            config = get_sanity_config();
        } else if (profile_str == "stress") {
            config = get_stress_config();
        } else if (profile_str == "benchmark") {
            config = get_benchmark_config();
        } else {
            std::cerr << "Error: Unknown profile '" << profile_str << "'\n";
            std::cerr << "Valid profiles: sanity, stress, benchmark\n";
            return 1;
        }

        // Apply command line overrides
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

        if (result.count("sizes")) {
            config.packet_sizes = parse_sizes(result["sizes"].as<std::string>());
        }

        if (result.count("packets")) {
            config.num_packets = result["packets"].as<uint32_t>();
        }

        if (result.count("noc-types")) {
            config.noc_types = split(result["noc-types"].as<std::string>(), ',');
        }

        // Category toggles
        if (result.count("disable-simple")) {
            config.categories.simple_unicast = false;
        }
        if (result.count("disable-inter-mesh")) {
            config.categories.inter_mesh = false;
        }
        if (result.count("disable-all-to-all")) {
            config.categories.all_to_all = false;
        }
        if (result.count("enable-random")) {
            config.categories.random_pairing = true;
        }
        if (result.count("enable-all-to-one")) {
            config.categories.all_to_one = true;
        }
        if (result.count("enable-flow-control")) {
            config.categories.flow_control = true;
        }
        if (result.count("enable-sequential")) {
            config.categories.sequential = true;
        }

        if (result.count("name-prefix")) {
            config.test_name_prefix = result["name-prefix"].as<std::string>();
        }

        if (result.count("skip")) {
            config.skip_platforms = split(result["skip"].as<std::string>(), ',');
        }

        if (result.count("no-sync")) {
            config.include_sync = false;
        }

        fs::path cabling_path = result["cabling-descriptor-path"].as<std::string>();
        fs::path output_path = result["output-path"].as<std::string>();
        bool verbose = result.count("verbose") > 0;

        generate_traffic_tests(cabling_path, output_path, config, verbose);

        std::cout << "Generated: " << output_path << "\n";
        if (config.generate_mgd && !config.mgd_output_path.empty()) {
            std::cout << "Generated MGD: " << config.mgd_output_path << "\n";
        }

        return 0;

    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
