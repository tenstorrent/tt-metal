// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <string>
#include "generate_mgd.hpp"

void print_usage() {
    std::cout << "Usage: generate_mgd --cabling-descriptor-path <path> [--output-path <path>]" << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  --cabling-descriptor-path: Path to the cabling descriptor textproto file (required)" << std::endl;
    std::cout << "  --output-path: Path to output MGD file (default: mesh_graph_descriptor.textproto)" << std::endl;
    std::cout << "  --help: Print this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  ./build/tools/scaleout/generate_mgd \\" << std::endl;
    std::cout
        << "      --cabling-descriptor-path tools/tests/scaleout/cabling_descriptors/16_n300_lb_cluster.textproto \\"
        << std::endl;
    std::cout << "      --output-path output_mgd.textproto" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string cabling_descriptor_path;
    std::string output_path = "mesh_graph_descriptor.textproto";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }

        if (arg == "--cabling-descriptor-path") {
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

    try {
        // Generate the MGD
        tt::scaleout_tools::generate_mesh_graph_descriptor(cabling_descriptor_path, output_path);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
