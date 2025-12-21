// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <string>
#include <optional>
#include <chrono>
#include <sstream>

#include <factory_system_descriptor/utils.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt-metalium/distributed.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include "tests/tt_metal/test_utils/test_common.hpp"
#include <cabling_generator/cabling_generator.hpp>
#include <tt-metalium/hal.hpp>
#include "tools/scaleout/fabric_manager/utils/fabric_manager_utils.hpp"
#include <enchantum/enchantum.hpp>

namespace tt::scaleout_tools {

// Captures current list of supported input args
struct InputArgs {
    std::filesystem::path output_path = "";
    bool initialize_fabric = false;
    bool terminate_fabric = false;
    bool help = false;
    std::optional<std::string> fabric_config = std::nullopt;
    std::optional<std::string> reliability_mode = std::nullopt;
    std::optional<uint8_t> num_routing_planes = std::nullopt;
    std::optional<std::string> fabric_tensix_config = std::nullopt;
    std::optional<std::string> fabric_udm_mode = std::nullopt;
    std::optional<std::string> mesh_shape = std::nullopt;
};

std::filesystem::path generate_output_dir() {
    const auto& rt_options = tt::tt_metal::MetalContext::instance().rtoptions();
    std::filesystem::path output_dir_path = rt_options.get_root_dir() + "fabric_manager_logs/";
    std::filesystem::create_directories(output_dir_path);
    return output_dir_path;
}

InputArgs parse_input_args(const std::vector<std::string>& args_vec) {
    InputArgs input_args;

    if (test_args::has_command_option(args_vec, "--output-path")) {
        input_args.output_path = std::filesystem::path(test_args::get_command_option(args_vec, "--output-path"));
    } else {
        input_args.output_path = generate_output_dir();
    }
    if (test_args::has_command_option(args_vec, "--fabric-config")) {
        input_args.fabric_config = test_args::get_command_option(args_vec, "--fabric-config");
    }
    if (test_args::has_command_option(args_vec, "--reliability-mode")) {
        input_args.reliability_mode = test_args::get_command_option(args_vec, "--reliability-mode");
    }
    if (test_args::has_command_option(args_vec, "--num-routing-planes")) {
        input_args.num_routing_planes = std::stoi(test_args::get_command_option(args_vec, "--num-routing-planes"));
    }
    if (test_args::has_command_option(args_vec, "--fabric-tensix-config")) {
        input_args.fabric_tensix_config = test_args::get_command_option(args_vec, "--fabric-tensix-config");
    }
    if (test_args::has_command_option(args_vec, "--fabric-udm-mode")) {
        input_args.fabric_udm_mode = test_args::get_command_option(args_vec, "--fabric-udm-mode");
    }
    if (test_args::has_command_option(args_vec, "--mesh-shape")) {
        input_args.mesh_shape = test_args::get_command_option(args_vec, "--mesh-shape");
    }
    log_output_rank("Generating Fabric Management Logs in " + input_args.output_path.string());

    input_args.initialize_fabric = test_args::has_command_option(args_vec, "--initialize-fabric");
    input_args.terminate_fabric = test_args::has_command_option(args_vec, "--terminate-fabric");
    input_args.help = test_args::has_command_option(args_vec, "--help");

    TT_FATAL(
        !(input_args.initialize_fabric && input_args.terminate_fabric),
        "Cannot specify both --initialize-fabric and --terminate-fabric simultaneously");

    return input_args;
}

void print_usage_info() {
    std::cout << "Utility to manage Fabric Configuration and Routing for a Multi-Node TT Cluster" << std::endl;
    std::cout << "Provides fabric management capabilities including initialization and termination" << std::endl
              << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  --output-path: Path to output directory" << std::endl;
    std::cout << "  --fabric-config: Fabric configuration mode (FABRIC_1D, FABRIC_2D, etc.)" << std::endl;
    std::cout << "  --reliability-mode: Fabric reliability mode (STRICT_SYSTEM_HEALTH_SETUP_MODE, etc.)" << std::endl;
    std::cout << "  --num-routing-planes: Number of routing planes" << std::endl;
    std::cout << "  --fabric-tensix-config: Fabric tensix configuration (DISABLED, MUX)" << std::endl;
    std::cout << "  --fabric-udm-mode: Fabric UDM mode (DISABLED, ENABLED)" << std::endl;
    std::cout << "  --initialize-fabric: Initialize fabric configuration" << std::endl;
    std::cout << "  --mesh-shape: Mesh shape in ROWxCOL or ROW,COL format (e.g., 8x4 or 8,4)" << std::endl;
    std::cout << "  --terminate-fabric: Terminate fabric configuration" << std::endl;
    std::cout << "  --help: Print usage information" << std::endl << std::endl;
    std::cout << "To run on a multi-node cluster, use mpirun with a --hostfile option" << std::endl;
}

void set_config_vars() {
    // This tool must be run with slow dispatch mode, since
    // it manages fabric configuration which shouldn't
    // be running fabric routers during configuration
    setenv("TT_METAL_SLOW_DISPATCH_MODE", "1", 1);
    // Set env vars required by Control Plane when running on a multi-node cluster
    setenv("TT_MESH_HOST_RANK", "0", 1);
    setenv("TT_MESH_ID", "0", 1);
}

}  // namespace tt::scaleout_tools

int main(int argc, char* argv[]) {
    using namespace tt::scaleout_tools;

    set_config_vars();

    auto input_args = parse_input_args(std::vector<std::string>(argv, argv + argc));
    if (input_args.help) {
        print_usage_info();
        return 0;
    }

    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();

    // Parse enums and mesh shape if fabric operations are requested
    if (input_args.initialize_fabric || input_args.terminate_fabric) {
        TT_FATAL(
            input_args.mesh_shape.has_value(),
            "--mesh-shape is required when using --initialize-fabric or --terminate-fabric");

        // Parse enum values
        std::optional<tt::tt_fabric::FabricConfig> fabric_config_enum = std::nullopt;
        if (input_args.fabric_config.has_value()) {
            fabric_config_enum = enchantum::cast<tt::tt_fabric::FabricConfig>(input_args.fabric_config.value());
        }

        std::optional<tt::tt_fabric::FabricReliabilityMode> reliability_mode_enum = std::nullopt;
        if (input_args.reliability_mode.has_value()) {
            reliability_mode_enum =
                enchantum::cast<tt::tt_fabric::FabricReliabilityMode>(input_args.reliability_mode.value());
        }

        std::optional<tt::tt_fabric::FabricTensixConfig> fabric_tensix_config_enum = std::nullopt;
        if (input_args.fabric_tensix_config.has_value()) {
            fabric_tensix_config_enum =
                enchantum::cast<tt::tt_fabric::FabricTensixConfig>(input_args.fabric_tensix_config.value());
        }

        std::optional<tt::tt_fabric::FabricUDMMode> fabric_udm_mode_enum = std::nullopt;
        if (input_args.fabric_udm_mode.has_value()) {
            fabric_udm_mode_enum = enchantum::cast<tt::tt_fabric::FabricUDMMode>(input_args.fabric_udm_mode.value());
        }

        // Parse mesh shape
        std::string mesh_shape_str = input_args.mesh_shape.value();
        size_t delimiter_pos = mesh_shape_str.find('x');
        if (delimiter_pos == std::string::npos) {
            delimiter_pos = mesh_shape_str.find(',');
        }
        TT_FATAL(
            delimiter_pos != std::string::npos,
            "Invalid mesh shape format: {}. Expected format: ROWxCOL or ROW,COL (e.g., 8x4 or 8,4)",
            mesh_shape_str);
        uint32_t mesh_width;
        uint32_t mesh_height;
        try {
            mesh_width = std::stoul(mesh_shape_str.substr(0, delimiter_pos));
            mesh_height = std::stoul(mesh_shape_str.substr(delimiter_pos + 1));
        } catch (const std::exception& e) {
            TT_FATAL(false, "Failed to parse mesh shape: {}. Error: {}", mesh_shape_str, e.what());
        }
        tt::tt_metal::distributed::MeshShape mesh_shape(mesh_width, mesh_height);

        // Configure fabric if requested
        if (input_args.initialize_fabric) {
            configure_fabric_routing(
                fabric_config_enum,
                reliability_mode_enum,
                input_args.num_routing_planes,
                fabric_tensix_config_enum,
                fabric_udm_mode_enum,
                tt::tt_fabric::FabricManagerMode::INIT_FABRIC,
                mesh_shape,
                input_args.output_path);
        }

        if (input_args.terminate_fabric) {
            configure_fabric_routing(
                fabric_config_enum,
                reliability_mode_enum,
                input_args.num_routing_planes,
                fabric_tensix_config_enum,
                fabric_udm_mode_enum,
                tt::tt_fabric::FabricManagerMode::TERMINATE_FABRIC,
                mesh_shape,
                input_args.output_path);
        }
    }

    distributed_context.barrier();
    return 0;
}
