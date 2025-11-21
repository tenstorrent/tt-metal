// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <string>
#include <algorithm>
#include <cxxopts.hpp>
#include "generate_mgd.hpp"

int main(int argc, char* argv[]) {
    cxxopts::Options options(
        "generate_mgd", "Generate Mesh Graph Descriptors from cabling descriptors for fabric testing");

    options.add_options()(
        "c,cabling-descriptor-path", "Path to the cabling descriptor textproto file", cxxopts::value<std::string>())(
        "o,output-path",
        "Path to output MGD file",
        cxxopts::value<std::string>()->default_value("mesh_graph_descriptor.textproto"))(
        "f,format", "Output format: 'textproto' or 'yaml'", cxxopts::value<std::string>()->default_value("textproto"))(
        "v,verbose", "Enable verbose output", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))(
        "h,help", "Print usage information");

    try {
        auto result = options.parse(argc, argv);

        if (result.count("help") || argc == 1) {
            std::cout << options.help() << std::endl;
            return 0;
        }

        if (!result.count("cabling-descriptor-path")) {
            throw std::invalid_argument("--cabling-descriptor-path is required");
        }

        // Get arguments
        std::string cabling_descriptor_path = result["cabling-descriptor-path"].as<std::string>();
        std::string output_path = result["output-path"].as<std::string>();
        std::string format_str = result["format"].as<std::string>();
        bool verbose = result["verbose"].as<bool>();

        // Convert format string to lowercase for case-insensitive comparison
        std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::tolower);

        // Parse format
        tt::scaleout_tools::OutputFormat format;
        if (format_str == "textproto") {
            format = tt::scaleout_tools::OutputFormat::TEXTPROTO;
        } else if (format_str == "yaml") {
            format = tt::scaleout_tools::OutputFormat::YAML;
        } else {
            throw std::invalid_argument("Unknown format '" + format_str + "'. Supported formats: textproto, yaml");
        }

        // Generate the MGD
        tt::scaleout_tools::generate_mesh_graph_descriptor(cabling_descriptor_path, output_path, format, verbose);
        return 0;

    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        std::cerr << "Use --help for usage information" << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
