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

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " <cluster_descriptor_path> <deployment_descriptor_path> <output_name> [--simple]" << std::endl;
    std::cerr << "       " << program_name << " --help" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Arguments:" << std::endl;
    std::cerr << "  cluster_descriptor_path:    Path to the cluster descriptor file (.textproto)" << std::endl;
    std::cerr << "  deployment_descriptor_path: Path to the deployment descriptor file (.textproto)" << std::endl;
    std::cerr << "  output_name:                Name suffix for output files (without extension)" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --simple                    Generate simple CSV output (hostname-based) instead of detailed" << std::endl;
    std::cerr << "                              location information (rack, shelf, etc.)" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Output files:" << std::endl;
    std::cerr << "  - out/scaleout/factory_system_descriptor_<output_name>.textproto" << std::endl;
    std::cerr << "  - out/scaleout/cabling_guide_<output_name>.csv" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Examples:" << std::endl;
    std::cerr << "  " << program_name << " cluster.textproto deployment.textproto test" << std::endl;
    std::cerr << "  # Generates detailed CSV with rack/shelf information" << std::endl;
    std::cerr << "  " << program_name << " cluster.textproto deployment.textproto test --simple" << std::endl;
    std::cerr << "  # Generates simple CSV with hostname information only" << std::endl;
}

bool file_exists(const std::string& path) {
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

InputConfig parse_arguments(int argc, char** argv) {
    // Handle help flag
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        print_usage(argv[0]);
        exit(0);
    }
    
    // Validate argument count (3 required + optional --simple flag)
    if (argc < 4 || argc > 5) {
        std::cerr << "Error: Expected 3 arguments with optional --simple flag, got " << (argc - 1) << std::endl;
        print_usage(argv[0]);
        exit(1);
    }

    InputConfig config;
    config.cluster_descriptor_path = argv[1];
    config.deployment_descriptor_path = argv[2];
    config.output_name = argv[3];
    config.loc_info = true;  // Default to detailed location info
    
    // Check for --simple flag
    if (argc == 5) {
        if (std::string(argv[4]) == "--simple") {
            config.loc_info = false;
        } else {
            throw std::invalid_argument("Unknown flag: '" + std::string(argv[4]) + "'. Only --simple is supported.");
        }
    }
    
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
    
    // Validate output name
    if (config.output_name.empty()) {
        throw std::invalid_argument("Output name cannot be empty");
    }
    
    // Check for invalid filename characters
    const std::string invalid_chars = "<>:\"/|?*";
    for (char c : config.output_name) {
        if (invalid_chars.find(c) != std::string::npos) {
            throw std::invalid_argument("Output name contains invalid character '" + std::string(1, c) + "'. Avoid: " + invalid_chars);
        }
    }
    
    return config;
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