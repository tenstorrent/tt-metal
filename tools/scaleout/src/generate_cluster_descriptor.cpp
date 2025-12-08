// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <enchantum/enchantum.hpp>
#include <fstream>
#include <filesystem>
#include <string>
#include <stdexcept>
#include <iostream>
#include <cxxopts.hpp>

#include <factory_system_descriptor/utils.hpp>

using namespace tt::scaleout_tools;

struct InputConfig {
    std::string fsd_path;
    std::string output_dir;
    std::string base_filename;
};

bool file_exists(const std::string& path) {
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

InputConfig parse_arguments(int argc, char** argv) {
    cxxopts::Options options(
        "generate_cluster_descriptor", "Generate cluster descriptor(s) from a Factory System Descriptor (FSD)");

    options.add_options()(
        "f,fsd", "Path to the Factory System Descriptor file (.textproto)", cxxopts::value<std::string>())(
        "o,output-dir",
        "Directory where cluster descriptor files will be written",
        cxxopts::value<std::string>()->default_value("out/scaleout"))(
        "b,base-filename",
        "Base name for generated files (without extensions)",
        cxxopts::value<std::string>()->default_value("cluster_desc"))("h,help", "Print usage information");

    try {
        auto result = options.parse(argc, argv);

        if (result.count("help") || argc == 1) {
            std::cout << options.help() << std::endl;
            std::cout << "\nDescription:" << std::endl;
            std::cout << "  This tool generates cluster descriptor YAML files from a Factory System Descriptor."
                      << std::endl;
            std::cout << "  It automatically detects single-host vs multi-host systems and generates appropriate files."
                      << std::endl;
            std::cout << "\nOutput files:" << std::endl;
            std::cout << "  Single-host system:" << std::endl;
            std::cout << "    - {output_dir}/{base_filename}.yaml" << std::endl;
            std::cout << "\n  Multi-host system:" << std::endl;
            std::cout << "    - {output_dir}/{base_filename}_rank_0.yaml" << std::endl;
            std::cout << "    - {output_dir}/{base_filename}_rank_1.yaml" << std::endl;
            std::cout << "    - ... (one per host)" << std::endl;
            std::cout << "    - {output_dir}/{base_filename}_mapping.yaml" << std::endl;
            std::cout << "\nExamples:" << std::endl;
            std::cout << "  " << argv[0] << " --fsd my_fsd.textproto" << std::endl;
            std::cout << "  # Generates cluster descriptor(s) in out/scaleout/ with default naming" << std::endl;
            std::cout << "\n  " << argv[0]
                      << " --fsd my_fsd.textproto --output-dir /tmp/cluster --base-filename my_cluster" << std::endl;
            std::cout << "  # Generates cluster descriptor(s) in /tmp/cluster/ with custom naming" << std::endl;
            exit(0);
        }

        if (!result.count("fsd")) {
            throw std::invalid_argument("Factory System Descriptor (FSD) path is required");
        }

        InputConfig config;
        config.fsd_path = result["fsd"].as<std::string>();
        config.output_dir = result["output-dir"].as<std::string>();
        config.base_filename = result["base-filename"].as<std::string>();

        // Validate FSD file
        if (!file_exists(config.fsd_path)) {
            throw std::invalid_argument("FSD file not found: '" + config.fsd_path + "'");
        }

        // Validate file extension
        if (!config.fsd_path.ends_with(".textproto")) {
            std::cerr << "Warning: FSD file should typically have .textproto extension: '" << config.fsd_path << "'"
                      << std::endl;
        }

        // Check for invalid filename characters in base_filename
        const std::string invalid_chars = "<>:\"/\\|?*";
        for (char c : config.base_filename) {
            if (invalid_chars.find(c) != std::string::npos) {
                throw std::invalid_argument(
                    "Base filename contains invalid character '" + std::string(1, c) + "'. Avoid: " + invalid_chars);
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

        std::cout << "Generating cluster descriptor(s) from FSD..." << std::endl;
        std::cout << "  FSD file: " << config.fsd_path << std::endl;
        std::cout << "  Output directory: " << config.output_dir << std::endl;
        std::cout << "  Base filename: " << config.base_filename << std::endl;

        // Ensure output directory exists
        std::filesystem::create_directories(config.output_dir);

        std::cout << "Processing FSD and generating cluster descriptor(s)..." << std::endl;
        std::string output_file =
            generate_cluster_descriptor_from_fsd(config.fsd_path, config.output_dir, config.base_filename);

        std::cout << "\nSuccessfully generated cluster descriptor(s)!" << std::endl;
        std::cout << "  Main output file: " << output_file << std::endl;

        // Check if this is a multi-host mapping file to provide additional info
        if (output_file.find("_mapping.yaml") != std::string::npos) {
            std::cout << "  (Multi-host system detected - individual rank files also generated)" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
