// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <enchantum/enchantum.hpp>
#include <fstream>
#include <filesystem>
#include <string>
#include <stdexcept>
#include <iostream>
#include <map>
#include <tuple>
#include <vector>
#include <google/protobuf/text_format.h>
#include <cxxopts.hpp>

#include <cabling_generator/cabling_generator.hpp>
#include <factory_system_descriptor/utils.hpp>
#include <node/node_types.hpp>

using namespace tt::scaleout_tools;

struct InputConfig {
    std::string cabling_descriptor_path;  // Can be file or directory
    std::string deployment_descriptor_path;
    std::string output_name;
    bool loc_info = true;  // Default to detailed location info
    bool is_cabling_directory = false;
    // Opt-in sub-cluster filter: each entry is a path of instance keys (e.g. {"bh_galaxy_sp_0"}).
    std::vector<std::vector<std::string>> include_paths;
    std::vector<std::vector<std::string>> exclude_paths;

    // Nested-aggregation mode: nest each child descriptor as a subgraph under a composite root,
    // preserving hierarchy in the emitted FSD/cabling descriptor (see merge_cluster_configs.py).
    bool aggregate = false;
    std::string composite_name;
    std::vector<CablingGenerator::AggregateChild> children;
    std::vector<std::string> glue_paths;
};

// Split a slash-delimited instance path (e.g. "bh_galaxy_sp_0/bh_galaxy_node_2") into its keys.
// Rejects empty segments (empty value, leading/trailing/double slash) rather than dropping them, so a
// malformed --include/--exclude value errors instead of silently widening the output.
std::vector<std::string> split_instance_path(const std::string& flag, const std::string& path) {
    std::vector<std::string> keys;
    size_t start = 0;
    while (true) {
        size_t slash = path.find('/', start);
        std::string key = (slash == std::string::npos) ? path.substr(start) : path.substr(start, slash - start);
        if (key.empty()) {
            throw std::invalid_argument("Invalid --" + flag + " path '" + path + "': empty path segment");
        }
        keys.push_back(std::move(key));
        if (slash == std::string::npos) {
            break;
        }
        start = slash + 1;
    }
    return keys;
}

bool file_exists(const std::string& path) {
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

bool directory_exists(const std::string& path) {
    return std::filesystem::exists(path) && std::filesystem::is_directory(path);
}

InputConfig parse_arguments(int argc, char** argv) {
    cxxopts::Options options(
        "run_cabling_generator",
        "Generate factory system descriptor and cabling guide from cabling and deployment descriptors.\n"
        "The cabling descriptor can be a single .textproto file or a directory containing multiple\n"
        ".textproto files that will be merged together.");

    options.add_options()(
        "c,cabling",
        "Path to cabling descriptor file (.textproto) or directory containing multiple descriptors",
        cxxopts::value<std::string>())(
        "d,deployment", "Path to the deployment descriptor file (.textproto)", cxxopts::value<std::string>())(
        "o,output",
        "Name suffix for output files (without extensions) - optional, defaults to empty",
        cxxopts::value<std::string>()->default_value(""))(
        "s,simple",
        "Generate simple CSV output (hostname-based) instead of detailed location information (rack, shelf, etc.)",
        cxxopts::value<bool>()->default_value("false")->implicit_value("true"))(
        "include",
        "Opt-in sub-cluster filter: slash-delimited instance path selecting a node or subgraph to keep "
        "(e.g. --include bh_galaxy_sp_0 --include bh_galaxy_sp_1/bh_galaxy_node_2). Repeatable. A path "
        "matches any instance whose full path ends with those segments. Only connections whose endpoints "
        "are both within the selection are emitted.",
        cxxopts::value<std::vector<std::string>>())(
        "exclude",
        "Opt-in sub-cluster filter: slash-delimited instance path to drop (e.g. --exclude bh_galaxy_node_0). "
        "Repeatable. Applied after --include (removes from the kept set; if no --include is given, keeps "
        "everything except the excluded paths).",
        cxxopts::value<std::vector<std::string>>())(
        "aggregate",
        "Nested aggregation mode: nest each --child descriptor as a subgraph under a composite root "
        "(named by --name), wiring --glue inter-child cabling at the composite level. Preserves hierarchy "
        "in the emitted FSD instance_path and cabling descriptor. Requires --name, --deployment, and one or "
        "more --child; --cabling is ignored.",
        cxxopts::value<bool>()->default_value("false")->implicit_value("true"))(
        "name",
        "Composite root name for --aggregate mode (becomes the first FSD instance_path segment).",
        cxxopts::value<std::string>())(
        "child",
        "Aggregation child as <instance_name>=<cabling_descriptor.textproto>. Repeatable.",
        cxxopts::value<std::vector<std::string>>())(
        "child-fsd",
        "Aggregation child FSD as <instance_name>=<factory_system_descriptor.textproto>, supplying the "
        "child's intra-cluster hierarchy (per-host instance_path). Repeatable; matched to --child by "
        "instance_name.",
        cxxopts::value<std::vector<std::string>>())(
        "child-deployment",
        "Aggregation child deployment as <instance_name>=<deployment_descriptor.textproto>, letting the "
        "child load against its own host_id-indexed deployment (required for hierarchical, "
        "instance-named children). Repeatable; matched to --child by instance_name.",
        cxxopts::value<std::vector<std::string>>())(
        "glue",
        "Aggregation glue (inter-child) cabling descriptor file. Repeatable.",
        cxxopts::value<std::vector<std::string>>())("h,help", "Print usage information");

    try {
        auto result = options.parse(argc, argv);

        if (result.contains("help") || argc == 1) {
            std::cout << options.help() << std::endl;
            std::cout << "\nOutput files:" << std::endl;
            std::cout << "  - out/scaleout/factory_system_descriptor_<output_name>.textproto" << std::endl;
            std::cout << "  - out/scaleout/cabling_guide_<output_name>.csv" << std::endl;
            std::cout << "\nExamples:" << std::endl;
            std::cout << "  " << argv[0] << " --cabling cabling.textproto --deployment deployment.textproto"
                      << std::endl;
            std::cout << "  # Generates files with default names (no suffix)" << std::endl;
            std::cout << std::endl;
            std::cout << "  " << argv[0]
                      << " --cabling ./cabling_descriptors/ --deployment deployment.textproto --output merged"
                      << std::endl;
            std::cout << "  # Merges all .textproto files in directory and generates merged output" << std::endl;
            std::cout << std::endl;
            std::cout << "  " << argv[0]
                      << " --cabling cabling.textproto --deployment deployment.textproto --output test --simple"
                      << std::endl;
            std::cout << "  # Generates simple CSV with hostname information only" << std::endl;
            exit(0);
        }

        InputConfig config;
        config.aggregate = result["aggregate"].as<bool>();

        if (!result.contains("deployment")) {
            throw std::invalid_argument("Deployment descriptor path is required");
        }
        config.deployment_descriptor_path = result["deployment"].as<std::string>();
        config.output_name = result["output"].as<std::string>();
        config.loc_info = !result["simple"].as<bool>();

        if (config.aggregate) {
            if (!result.contains("name")) {
                throw std::invalid_argument("--aggregate requires --name");
            }
            if (!result.contains("child")) {
                throw std::invalid_argument("--aggregate requires at least one --child");
            }
            config.composite_name = result["name"].as<std::string>();
            // Parse repeatable <instance_name>=<path> options into a name->path map with validation.
            auto parse_name_path_map = [&](const char* opt) -> std::map<std::string, std::string> {
                std::map<std::string, std::string> out;
                if (!result.contains(opt)) {
                    return out;
                }
                for (const auto& raw : result[opt].as<std::vector<std::string>>()) {
                    auto eq = raw.find('=');
                    if (eq == std::string::npos || eq == 0 || eq + 1 >= raw.size()) {
                        throw std::invalid_argument(
                            std::string("--") + opt + " must be <instance_name>=<path>: '" + raw + "'");
                    }
                    std::string name = raw.substr(0, eq);
                    std::string path = raw.substr(eq + 1);
                    if (!file_exists(path) || !path.ends_with(".textproto")) {
                        throw std::invalid_argument(
                            std::string("--") + opt + " descriptor not found or not .textproto: '" + path + "'");
                    }
                    out[std::move(name)] = std::move(path);
                }
                return out;
            };
            // Parse --child-fsd / --child-deployment first so each child can be paired by instance_name.
            std::map<std::string, std::string> child_fsd_by_name = parse_name_path_map("child-fsd");
            std::map<std::string, std::string> child_deployment_by_name = parse_name_path_map("child-deployment");
            for (const auto& raw : result["child"].as<std::vector<std::string>>()) {
                auto eq = raw.find('=');
                if (eq == std::string::npos || eq == 0 || eq + 1 >= raw.size()) {
                    throw std::invalid_argument("--child must be <instance_name>=<path>: '" + raw + "'");
                }
                std::string name = raw.substr(0, eq);
                std::string path = raw.substr(eq + 1);
                if (!file_exists(path) || !path.ends_with(".textproto")) {
                    throw std::invalid_argument(
                        "--child cabling descriptor not found or not .textproto: '" + path + "'");
                }
                CablingGenerator::AggregateChild ac;
                ac.cabling_path = std::move(path);
                if (auto it = child_fsd_by_name.find(name); it != child_fsd_by_name.end()) {
                    ac.fsd_path = it->second;
                }
                if (auto it = child_deployment_by_name.find(name); it != child_deployment_by_name.end()) {
                    ac.deployment_path = it->second;
                }
                ac.instance_name = std::move(name);
                config.children.push_back(std::move(ac));
            }
            if (result.contains("glue")) {
                for (const auto& path : result["glue"].as<std::vector<std::string>>()) {
                    if (!file_exists(path) || !path.ends_with(".textproto")) {
                        throw std::invalid_argument("--glue descriptor not found or not .textproto: '" + path + "'");
                    }
                    config.glue_paths.push_back(path);
                }
            }
            // Validate deployment file (shared with non-aggregate path below).
            if (!file_exists(config.deployment_descriptor_path)) {
                throw std::invalid_argument(
                    "Deployment descriptor file not found: '" + config.deployment_descriptor_path + "'");
            }
            if (!config.deployment_descriptor_path.ends_with(".textproto")) {
                throw std::invalid_argument(
                    "Deployment descriptor file should have .textproto extension: '" +
                    config.deployment_descriptor_path + "'");
            }
            if (!config.output_name.empty()) {
                const std::string invalid_chars = "<>:\"/|?*";
                for (char c : config.output_name) {
                    if (invalid_chars.find(c) != std::string::npos) {
                        throw std::invalid_argument(
                            "Output name contains invalid character '" + std::string(1, c) +
                            "'. Avoid: " + invalid_chars);
                    }
                }
                config.output_name = "_" + config.output_name;
            }
            return config;
        }

        if (!result.contains("cabling")) {
            throw std::invalid_argument("Cabling descriptor path is required");
        }

        config.cabling_descriptor_path = result["cabling"].as<std::string>();
        if (result.contains("include")) {
            for (const auto& raw_path : result["include"].as<std::vector<std::string>>()) {
                config.include_paths.push_back(split_instance_path("include", raw_path));
            }
        }
        if (result.contains("exclude")) {
            for (const auto& raw_path : result["exclude"].as<std::vector<std::string>>()) {
                config.exclude_paths.push_back(split_instance_path("exclude", raw_path));
            }
        }

        // Check if cabling descriptor is a directory or file
        if (directory_exists(config.cabling_descriptor_path)) {
            config.is_cabling_directory = true;
        } else if (file_exists(config.cabling_descriptor_path)) {
            config.is_cabling_directory = false;
            // Validate file extension for single file
            if (!config.cabling_descriptor_path.ends_with(".textproto")) {
                throw std::invalid_argument(
                    "Cabling descriptor file should have .textproto extension: '" + config.cabling_descriptor_path +
                    "'");
            }
        } else {
            throw std::invalid_argument(
                "Cabling descriptor path not found (expected file or directory): '" + config.cabling_descriptor_path +
                "'");
        }

        // Validate deployment descriptor file
        if (!file_exists(config.deployment_descriptor_path)) {
            throw std::invalid_argument(
                "Deployment descriptor file not found: '" + config.deployment_descriptor_path + "'");
        }

        if (!config.deployment_descriptor_path.ends_with(".textproto")) {
            throw std::invalid_argument(
                "Deployment descriptor file should have .textproto extension: '" + config.deployment_descriptor_path +
                "'");
        }

        // Check for invalid filename characters (only if output name is not empty)
        if (!config.output_name.empty()) {
            const std::string invalid_chars = "<>:\"/|?*";
            for (char c : config.output_name) {
                if (invalid_chars.find(c) != std::string::npos) {
                    throw std::invalid_argument(
                        "Output name contains invalid character '" + std::string(1, c) + "'. Avoid: " + invalid_chars);
                }
            }
            config.output_name = "_" + config.output_name;
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

        if (config.aggregate) {
            std::cout << "Aggregating cluster configuration (nested)..." << std::endl;
            std::cout << "  Composite name: " << config.composite_name << std::endl;
            std::cout << "  Deployment descriptor: " << config.deployment_descriptor_path << std::endl;
            for (const auto& child : config.children) {
                std::cout << "  + child: " << child.instance_name << " -> " << child.cabling_path;
                if (!child.fsd_path.empty()) {
                    std::cout << " (fsd: " << child.fsd_path << ")";
                }
                if (!child.deployment_path.empty()) {
                    std::cout << " (deployment: " << child.deployment_path << ")";
                }
                std::cout << std::endl;
            }
            for (const auto& g : config.glue_paths) {
                std::cout << "  + glue:  " << g << std::endl;
            }

            CablingGenerator cabling_generator = CablingGenerator::build_nested_aggregate(
                config.composite_name, config.children, config.glue_paths, config.deployment_descriptor_path);

            std::string factory_output = "out/scaleout/factory_system_descriptor" + config.output_name + ".textproto";
            std::string cabling_output = "out/scaleout/cabling_guide" + config.output_name + ".csv";
            std::string cabling_desc_output = "out/scaleout/cabling_descriptor" + config.output_name + ".textproto";
            std::string deployment_output = "out/scaleout/deployment_descriptor" + config.output_name + ".textproto";
            std::filesystem::create_directories("out/scaleout");

            std::cout << "Generating factory system descriptor..." << std::endl;
            cabling_generator.emit_factory_system_descriptor(factory_output);
            std::cout << "Generating cabling guide CSV..." << std::endl;
            cabling_generator.emit_cabling_guide_csv(cabling_output, config.loc_info);
            std::cout << "Generating hierarchical cabling descriptor..." << std::endl;
            cabling_generator.emit_cabling_descriptor(cabling_desc_output, /*hierarchical=*/true);
            std::cout << "Generating deployment descriptor..." << std::endl;
            cabling_generator.emit_deployment_descriptor(deployment_output);

            std::cout << "Successfully generated:" << std::endl;
            std::cout << "  - " << factory_output << std::endl;
            std::cout << "  - " << cabling_output << std::endl;
            std::cout << "  - " << cabling_desc_output << std::endl;
            std::cout << "  - " << deployment_output << std::endl;
            return 0;
        }

        std::cout << "Generating cabling configuration..." << std::endl;
        if (config.is_cabling_directory) {
            std::cout << "  Cabling descriptor directory: " << config.cabling_descriptor_path << std::endl;
            std::cout << "  (Will merge all .textproto files in directory)" << std::endl;
        } else {
            std::cout << "  Cabling descriptor: " << config.cabling_descriptor_path << std::endl;
        }
        std::cout << "  Deployment descriptor: " << config.deployment_descriptor_path << std::endl;
        std::cout << "  Output name suffix: " << config.output_name << std::endl;

        std::cout << "Loading descriptors and initializing generator..." << std::endl;
        auto print_paths = [](const char* label, const std::vector<std::vector<std::string>>& paths) {
            if (paths.empty()) {
                return;
            }
            std::cout << "  " << label << ":" << std::endl;
            for (const auto& path : paths) {
                std::string joined;
                for (const auto& key : path) {
                    joined += joined.empty() ? key : "/" + key;
                }
                std::cout << "    - " << joined << std::endl;
            }
        };
        print_paths("Include filter (sub-cluster)", config.include_paths);
        print_paths("Exclude filter (sub-cluster)", config.exclude_paths);
        CablingGenerator cabling_generator(config.cabling_descriptor_path, config.deployment_descriptor_path);
        if (!config.include_paths.empty() || !config.exclude_paths.empty()) {
            cabling_generator.apply_instance_filter(config.include_paths, config.exclude_paths);
        }

        std::string factory_output = "out/scaleout/factory_system_descriptor" + config.output_name + ".textproto";
        std::string cabling_output = "out/scaleout/cabling_guide" + config.output_name + ".csv";
        std::string cabling_desc_output = "out/scaleout/cabling_descriptor" + config.output_name + ".textproto";
        std::string deployment_output = "out/scaleout/deployment_descriptor" + config.output_name + ".textproto";

        // Ensure output directory exists
        std::filesystem::create_directories("out/scaleout");

        std::cout << "Generating factory system descriptor..." << std::endl;
        cabling_generator.emit_factory_system_descriptor(factory_output);

        std::cout << "Generating cabling guide CSV..." << std::endl;
        std::cout << "  CSV format: " << (config.loc_info ? "detailed (with location info)" : "simple (hostname-based)") << std::endl;
        cabling_generator.emit_cabling_guide_csv(cabling_output, config.loc_info);

        std::cout << "Generating merged cabling descriptor..." << std::endl;
        cabling_generator.emit_cabling_descriptor(cabling_desc_output);

        std::cout << "Generating deployment descriptor..." << std::endl;
        cabling_generator.emit_deployment_descriptor(deployment_output);

        std::cout << "Successfully generated:" << std::endl;
        std::cout << "  - " << factory_output << std::endl;
        std::cout << "  - " << cabling_output << std::endl;
        std::cout << "  - " << cabling_desc_output << std::endl;
        std::cout << "  - " << deployment_output << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
