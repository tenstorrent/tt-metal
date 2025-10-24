// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <enchantum/enchantum.hpp>
#include <fstream>
#include <filesystem>
#include <string>
#include <stdexcept>
#include <iostream>
#include <google/protobuf/text_format.h>
#include <cxxopts.hpp>

#include <cabling_generator/cabling_generator.hpp>
#include <factory_system_descriptor/utils.hpp>
#include <node/node_types.hpp>

using namespace tt::scaleout_tools;

struct InputConfig {
    std::string cluster_descriptor_path;
    std::string deployment_descriptor_path;
    std::string output_name;
    bool loc_info = true;  // Default to detailed location info
};

bool file_exists(const std::string& path) {
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

InputConfig parse_arguments(int argc, char** argv) {
    cxxopts::Options options("run_cabling_generator", "Generate factory system descriptor and cabling guide from cluster and deployment descriptors");

    options.add_options()
        ("c,cluster", "Path to the cluster descriptor file (.textproto)", cxxopts::value<std::string>())
        ("d,deployment", "Path to the deployment descriptor file (.textproto)", cxxopts::value<std::string>())
        ("o,output", "Name suffix for output files (without extensions) - optional, defaults to empty", cxxopts::value<std::string>()->default_value(""))
        ("s,simple", "Generate simple CSV output (hostname-based) instead of detailed location information (rack, shelf, etc.)", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("h,help", "Print usage information");

    try {
        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            std::cout << "\nOutput files:" << std::endl;
            std::cout << "  - out/scaleout/factory_system_descriptor_<output_name>.textproto" << std::endl;
            std::cout << "  - out/scaleout/cabling_guide_<output_name>.csv" << std::endl;
            std::cout << "\nExamples:" << std::endl;
            std::cout << "  " << argv[0] << " --cluster cluster.textproto --deployment deployment.textproto" << std::endl;
            std::cout << "  # Generates files with default names (no suffix)" << std::endl;
            std::cout << "  " << argv[0] << " --cluster cluster.textproto --deployment deployment.textproto --output test" << std::endl;
            std::cout << "  # Generates detailed CSV with rack/shelf information" << std::endl;
            std::cout << "  " << argv[0] << " --cluster cluster.textproto --deployment deployment.textproto --output test --simple" << std::endl;
            std::cout << "  # Generates simple CSV with hostname information only" << std::endl;
            exit(0);
        }

        if (!result.count("cluster")) {
            throw std::invalid_argument("Cluster descriptor path is required");
        }

        if (!result.count("deployment")) {
            throw std::invalid_argument("Deployment descriptor path is required");
        }

        InputConfig config;
        config.cluster_descriptor_path = result["cluster"].as<std::string>();
        config.deployment_descriptor_path = result["deployment"].as<std::string>();
        config.output_name = result["output"].as<std::string>();
        config.loc_info = !result["simple"].as<bool>();

        // Validate cluster descriptor file
        if (!file_exists(config.cluster_descriptor_path)) {
            throw std::invalid_argument("Cluster descriptor file not found: '" + config.cluster_descriptor_path + "'");
        }

        // Validate deployment descriptor file
        if (!file_exists(config.deployment_descriptor_path)) {
            throw std::invalid_argument("Deployment descriptor file not found: '" + config.deployment_descriptor_path + "'");
        }

        // Validate file extensions
        if (!config.cluster_descriptor_path.ends_with(".textproto")) {
            throw std::invalid_argument("Cluster descriptor file should have .textproto extension: '" + config.cluster_descriptor_path + "'");
        }

        if (!config.deployment_descriptor_path.ends_with(".textproto")) {
            throw std::invalid_argument("Deployment descriptor file should have .textproto extension: '" + config.deployment_descriptor_path + "'");
        }

        // Check for invalid filename characters (only if output name is not empty)
        if (!config.output_name.empty()) {
            const std::string invalid_chars = "<>:\"/|?*";
            for (char c : config.output_name) {
                if (invalid_chars.find(c) != std::string::npos) {
                    throw std::invalid_argument("Output name contains invalid character '" + std::string(1, c) + "'. Avoid: " + invalid_chars);
                }
            }
        }

        return config;

    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        exit(1);
    }
}

int main(int argc, char** argv) {
    try {
        InputConfig config = parse_arguments(argc, argv);

        std::cout << "Generating cabling configuration..." << std::endl;
        std::cout << "  Cluster descriptor: " << config.cluster_descriptor_path << std::endl;
        std::cout << "  Deployment descriptor: " << config.deployment_descriptor_path << std::endl;
        std::cout << "  Output name suffix: " << config.output_name << std::endl;

        std::cout << "Loading descriptors and initializing generator..." << std::endl;
        CablingGenerator cabling_generator(config.cluster_descriptor_path, config.deployment_descriptor_path);

        std::string factory_output = "out/scaleout/factory_system_descriptor_" + config.output_name + ".textproto";
        std::string cabling_output = "out/scaleout/cabling_guide_" + config.output_name + ".csv";

        // Ensure output directory exists
        std::filesystem::create_directories("out/scaleout");

        std::cout << "Generating factory system descriptor..." << std::endl;
        cabling_generator.emit_factory_system_descriptor(factory_output);

        std::cout << "Generating cabling guide CSV..." << std::endl;
        std::cout << "  CSV format: " << (config.loc_info ? "detailed (with location info)" : "simple (hostname-based)") << std::endl;
        cabling_generator.emit_cabling_guide_csv(cabling_output, config.loc_info);

        std::cout << "Successfully generated:" << std::endl;
        std::cout << "  - " << factory_output << std::endl;
        std::cout << "  - " << cabling_output << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
