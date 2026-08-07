// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <cxxopts.hpp>

#include <cabling_generator/regen_descriptors.hpp>
#include <factory_system_descriptor/subset_check.hpp>

using namespace tt::scaleout_tools;

namespace {

// Exit codes (stable for CI/automation gating).
constexpr int kExitOk = 0;             // regen succeeded (and skinny subset satisfied, if checked)
constexpr int kExitSkinnyMissing = 2;  // regen succeeded but the regenerated topology lacks skinny connections
constexpr int kExitError = 1;          // bad input / parse failure

struct InputConfig {
    std::string cabling_descriptor_path;
    std::string deployment_descriptor_path;
    std::string unretrainable_channels_path;
    std::string output_dir;
    std::string skinny_fsd_path;  // optional
};

bool file_exists(const std::string& path) {
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

bool directory_exists(const std::string& path) {
    return std::filesystem::exists(path) && std::filesystem::is_directory(path);
}

InputConfig parse_arguments(int argc, char** argv) {
    cxxopts::Options options(
        "run_regen_descriptors",
        "Regenerate FSD, cabling and deployment descriptors with unretrainable cables removed.\n"
        "\n"
        "Reads the unretrainable_channels.yaml artifact emitted by run_cluster_validation when\n"
        "it exhausts its retrain budget, then rebuilds a self-contained descriptor set that\n"
        "omits every cable whose channel expansion intersects the unretrainable set\n"
        "(dead-channel == dead-cable policy).\n"
        "\n"
        "Outputs three files under <output-dir>:\n"
        "  - factory_system_descriptor.textproto\n"
        "  - cabling_descriptor.textproto\n"
        "  - deployment_descriptor.textproto  (unchanged; cable failures don't move hardware)\n"
        "\n"
        "The original input descriptors are not modified.");

    options.add_options()(
        "c,cabling",
        "Path to original cabling descriptor file (.textproto) or directory of descriptors",
        cxxopts::value<std::string>())(
        "d,deployment", "Path to original deployment descriptor (.textproto)", cxxopts::value<std::string>())(
        "u,unretrainable-channels",
        "Path to unretrainable_channels.yaml emitted by run_cluster_validation",
        cxxopts::value<std::string>())(
        "o,output-dir", "Directory to write the regenerated descriptor set into", cxxopts::value<std::string>())(
        "s,skinny-fsd",
        "Optional: skinny (minimum required) FSD (.textproto). After regeneration, check whether the\n"
        "regenerated topology still contains every connection it lists; if not, the missing ones are\n"
        "reported and the tool exits 2.",
        cxxopts::value<std::string>())("h,help", "Print usage information");

    try {
        auto result = options.parse(argc, argv);

        if (result.contains("help") || argc == 1) {
            std::cout << options.help() << std::endl;
            std::cout << "\nExample:\n";
            std::cout
                << "  " << argv[0] << " --cabling /data/scaleout_configs/<cluster>/cabling_descriptor.textproto \\\n"
                << "             --deployment /data/scaleout_configs/<cluster>/deployment_descriptor.textproto \\\n"
                << "             --unretrainable-channels validation_output/unretrainable_channels.yaml \\\n"
                << "             --output-dir validation_output/regenerated/\n";
            std::exit(0);
        }

        for (const char* required : {"cabling", "deployment", "unretrainable-channels", "output-dir"}) {
            if (!result.contains(required)) {
                throw std::invalid_argument(std::string("--") + required + " is required");
            }
        }

        InputConfig config{
            .cabling_descriptor_path = result["cabling"].as<std::string>(),
            .deployment_descriptor_path = result["deployment"].as<std::string>(),
            .unretrainable_channels_path = result["unretrainable-channels"].as<std::string>(),
            .output_dir = result["output-dir"].as<std::string>(),
            .skinny_fsd_path = result.contains("skinny-fsd") ? result["skinny-fsd"].as<std::string>() : "",
        };

        if (!file_exists(config.cabling_descriptor_path) && !directory_exists(config.cabling_descriptor_path)) {
            throw std::invalid_argument(
                "Cabling descriptor not found (expected file or directory): '" + config.cabling_descriptor_path + "'");
        }
        if (!file_exists(config.deployment_descriptor_path)) {
            throw std::invalid_argument("Deployment descriptor not found: '" + config.deployment_descriptor_path + "'");
        }
        if (!file_exists(config.unretrainable_channels_path)) {
            throw std::invalid_argument(
                "Unretrainable channels YAML not found: '" + config.unretrainable_channels_path + "'");
        }
        if (!config.skinny_fsd_path.empty() && !file_exists(config.skinny_fsd_path)) {
            throw std::invalid_argument("Skinny FSD not found: '" + config.skinny_fsd_path + "'");
        }

        return config;
    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        std::exit(kExitError);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        auto config = parse_arguments(argc, argv);

        std::cout << "Regenerating descriptors with unretrainable cables removed..." << std::endl;
        std::cout << "  Cabling:                 " << config.cabling_descriptor_path << std::endl;
        std::cout << "  Deployment:              " << config.deployment_descriptor_path << std::endl;
        std::cout << "  Unretrainable channels:  " << config.unretrainable_channels_path << std::endl;
        std::cout << "  Output dir:              " << config.output_dir << std::endl;

        auto summary = regenerate_descriptors_excluding_dead_channels(
            config.cabling_descriptor_path,
            config.deployment_descriptor_path,
            config.unretrainable_channels_path,
            config.output_dir);

        std::cout << "\nDone." << std::endl;
        std::cout << "  Input dead channel endpoints: " << summary.input_dead_channels << std::endl;
        std::cout << "  Cables pruned from FSD:       " << summary.pruned_cables.size() << std::endl;
        std::cout << "  Channels in regenerated FSD:  " << summary.channels_remaining << std::endl;
        std::cout << "\n  Wrote:" << std::endl;
        std::cout << "    " << summary.output_fsd_path << std::endl;
        std::cout << "    " << summary.output_cabling_descriptor_path << std::endl;
        std::cout << "    " << summary.output_deployment_descriptor_path << std::endl;

        if (!summary.pruned_cables.empty()) {
            std::cout << "\nPruned cables (removed from FSD; inter-node ones also from cabling descriptor):"
                      << std::endl;
            for (const auto& [endpoint_a, endpoint_b] : summary.pruned_cables) {
                std::cout << "  - " << endpoint_a << " <-> " << endpoint_b << std::endl;
            }
        }

        // Optional: confirm the regenerated topology still satisfies a skinny (minimum) topology.
        if (!config.skinny_fsd_path.empty()) {
            auto missing = missing_skinny_connections(config.skinny_fsd_path, summary.output_fsd_path);
            std::cout << "\nSkinny check against " << config.skinny_fsd_path << ":" << std::endl;
            if (missing.empty()) {
                std::cout << "  PASS: regenerated topology contains the full skinny topology; safe to deploy on skinny."
                          << std::endl;
            } else {
                std::cerr << "  FAIL: " << missing.size()
                          << " skinny-required connection(s) missing from the regenerated topology:" << std::endl;
                for (const auto& [endpoint_a, endpoint_b] : missing) {
                    std::cerr << "    - " << endpoint_a << " <-> " << endpoint_b << std::endl;
                }
                return kExitSkinnyMissing;
            }
        }

        return kExitOk;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return kExitError;
    }
}
