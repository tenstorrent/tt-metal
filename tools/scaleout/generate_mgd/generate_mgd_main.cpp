// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <cxxopts.hpp>

#include "generate_mgd.hpp"

int main(int argc, char* argv[]) {
    cxxopts::Options options(
        "generate_mgd", "Generate Mesh Graph Descriptors from cabling descriptors for fabric testing");

    options.add_options()(
        "c,cabling-descriptor-path", "Path to the cabling descriptor textproto file", cxxopts::value<std::string>())(
        "o,output-path",
        "Path to output MGD textproto file",
        cxxopts::value<std::string>()->default_value("mesh_graph_descriptor.textproto"))(
        "v,verbose", "Enable verbose output", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))(
        "h,help", "Print usage information");

    try {
        const auto result = options.parse(argc, argv);

        if (result.count("help") || argc == 1) {
            std::cout << options.help() << '\n';
            return 0;
        }

        if (!result.count("cabling-descriptor-path")) {
            throw std::invalid_argument("--cabling-descriptor-path is required");
        }

        const std::filesystem::path cabling_descriptor_path = result["cabling-descriptor-path"].as<std::string>();
        const std::filesystem::path output_path = result["output-path"].as<std::string>();
        const bool verbose = result["verbose"].as<bool>();

        tt::scaleout_tools::generate_mesh_graph_descriptor(cabling_descriptor_path, output_path, verbose);
        return 0;

    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << '\n';
        std::cerr << "Use --help for usage information\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
