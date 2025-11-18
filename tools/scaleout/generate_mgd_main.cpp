// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <string>
#include "generate_mgd.hpp"

void print_usage() {
    std::cout << "Usage: generate_mgd --cabling-descriptor-path <path> [OPTIONS]" << std::endl;
    std::cout << std::endl;
    std::cout << "Generate Mesh Graph Descriptors from cabling descriptors for fabric testing." << std::endl;
    std::cout << std::endl;
    std::cout << "Required Arguments:" << std::endl;
    std::cout << "  --cabling-descriptor-path PATH" << std::endl;
    std::cout << "        Path to the cabling descriptor textproto file" << std::endl;
    std::cout << std::endl;
    std::cout << "Optional Arguments:" << std::endl;
    std::cout << "  --output-path PATH" << std::endl;
    std::cout << "        Path to output MGD file (default: mesh_graph_descriptor.textproto)" << std::endl;
    std::cout << "  --format FORMAT" << std::endl;
    std::cout << "        Output format: 'textproto' or 'yaml' (default: textproto)" << std::endl;
    std::cout << "  --verbose, -v" << std::endl;
    std::cout << "        Enable verbose output" << std::endl;
    std::cout << "  --help, -h" << std::endl;
    std::cout << "        Print this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  # Generate MGD for 16 N300 LB cluster" << std::endl;
    std::cout << "  ./build/tools/scaleout/generate_mgd \\" << std::endl;
    std::cout << "      --cabling-descriptor-path tools/tests/scaleout/cabling_descriptors/16_n300_lb_cluster.textproto"
              << std::endl;
    std::cout << std::endl;
    std::cout << "  # Generate YAML format with verbose output" << std::endl;
    std::cout << "  ./build/tools/scaleout/generate_mgd \\" << std::endl;
    std::cout << "      --cabling-descriptor-path path/to/cabling.textproto \\" << std::endl;
    std::cout << "      --output-path output.yaml --format yaml --verbose" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string cabling_descriptor_path;
    std::string output_path = "mesh_graph_descriptor.textproto";
    std::string format_str = "textproto";
    bool verbose = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }

        if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--cabling-descriptor-path") {
            if (i + 1 < argc) {
                cabling_descriptor_path = argv[++i];
            } else {
                std::cerr << "Error: --cabling-descriptor-path requires a value" << std::endl;
                print_usage();
                return 1;
            }
        } else if (arg == "--output-path") {
            if (i + 1 < argc) {
                output_path = argv[++i];
            } else {
                std::cerr << "Error: --output-path requires a value" << std::endl;
                print_usage();
                return 1;
            }
        } else if (arg == "--format") {
            if (i + 1 < argc) {
                format_str = argv[++i];
                // Convert to lowercase for case-insensitive comparison
                for (char& c : format_str) {
                    c = std::tolower(c);
                }
            } else {
                std::cerr << "Error: --format requires a value" << std::endl;
                print_usage();
                return 1;
            }
        } else {
            std::cerr << "Error: Unknown argument: " << arg << std::endl;
            print_usage();
            return 1;
        }
    }

    // Validate required arguments
    if (cabling_descriptor_path.empty()) {
        std::cerr << "Error: --cabling-descriptor-path is required" << std::endl;
        print_usage();
        return 1;
    }

    // Parse format
    tt::scaleout_tools::OutputFormat format;
    if (format_str == "textproto") {
        format = tt::scaleout_tools::OutputFormat::TEXTPROTO;
    } else if (format_str == "yaml") {
        format = tt::scaleout_tools::OutputFormat::YAML;
    } else {
        std::cerr << "Error: Unknown format '" << format_str << "'. Supported formats: textproto, yaml" << std::endl;
        return 1;
    }

    try {
        // Generate the MGD
        tt::scaleout_tools::generate_mesh_graph_descriptor(cabling_descriptor_path, output_path, format, verbose);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
