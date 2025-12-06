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
            std::cout << options.help() << '\n';
            std::cout << "\nDescription:" << '\n';
            std::cout << "  This tool generates cluster descriptor YAML files from a Factory System Descriptor."
                      << '\n';
            std::cout << "  It automatically detects single-host vs multi-host systems and generates appropriate files."
                      << '\n';
            std::cout << "\nOutput files:" << '\n';
            std::cout << "  Single-host system:" << '\n';
            std::cout << "    - {output_dir}/{base_filename}.yaml" << '\n';
            std::cout << "\n  Multi-host system:" << '\n';
            std::cout << "    - {output_dir}/{base_filename}_rank_0.yaml" << '\n';
            std::cout << "    - {output_dir}/{base_filename}_rank_1.yaml" << '\n';
            std::cout << "    - ... (one per host)" << '\n';
            std::cout << "    - {output_dir}/{base_filename}_mapping.yaml" << '\n';
            std::cout << "\nExamples:" << '\n';
            std::cout << "  " << argv[0] << " --fsd my_fsd.textproto" << '\n';
            std::cout << "  # Generates cluster descriptor(s) in out/scaleout/ with default naming" << '\n';
            std::cout << "\n  " << argv[0]
                      << " --fsd my_fsd.textproto --output-dir /tmp/cluster --base-filename my_cluster" << '\n';
            std::cout << "  # Generates cluster descriptor(s) in /tmp/cluster/ with custom naming" << '\n';
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
                      << '\n';
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
        std::cerr << "Error parsing arguments: " << e.what() << '\n';
        std::cerr << options.help() << '\n';
        exit(1);
    }
}

int main(int argc, char** argv) {
    try {
        InputConfig config = parse_arguments(argc, argv);

        std::cout << "Generating cluster descriptor(s) from FSD..." << '\n';
        std::cout << "  FSD file: " << config.fsd_path << '\n';
        std::cout << "  Output directory: " << config.output_dir << '\n';
        std::cout << "  Base filename: " << config.base_filename << '\n';

        // Ensure output directory exists
        std::filesystem::create_directories(config.output_dir);

        std::cout << "Processing FSD and generating cluster descriptor(s)..." << '\n';
        std::string output_file =
            generate_cluster_descriptor_from_fsd(config.fsd_path, config.output_dir, config.base_filename);

        std::cout << "\nSuccessfully generated cluster descriptor(s)!" << '\n';
        std::cout << "  Main output file: " << output_file << '\n';

        // Check if this is a multi-host mapping file to provide additional info
        if (output_file.find("_mapping.yaml") != std::string::npos) {
            std::cout << "  (Multi-host system detected - individual rank files also generated)" << '\n';
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
