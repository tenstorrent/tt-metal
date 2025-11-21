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

namespace tt::scaleout_tools {

using tt::tt_metal::PhysicalSystemDescriptor;

// Captures current list of supported input args
struct InputArgs {
    std::optional<std::string> cabling_descriptor_path = std::nullopt;
    std::optional<std::string> deployment_descriptor_path = std::nullopt;
    std::optional<std::string> fsd_path = std::nullopt;
    std::optional<std::string> gsd_path = std::nullopt;
    std::filesystem::path output_path = "";
    bool fail_on_warning = false;
    bool print_routing_tables = false;
    bool print_ethernet_channels = false;
    bool print_active_connections = false;
    bool print_all_connections = false;
    bool configure_fabric = false;
    bool initialize_fabric = false;
    bool help = false;
    std::optional<std::string> fabric_config = std::nullopt;
    std::optional<std::string> reliability_mode = std::nullopt;
    std::optional<uint8_t> num_routing_planes = std::nullopt;
    std::optional<std::string> fabric_tensix_config = std::nullopt;
};

std::filesystem::path generate_output_dir() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    std::string dir_name = ss.str();
    const auto& rt_options = tt::tt_metal::MetalContext::instance().rtoptions();
    std::filesystem::path output_dir_path = rt_options.get_root_dir() + "fabric_manager_logs/" + dir_name;
    std::filesystem::create_directories(output_dir_path);
    return output_dir_path;
}

InputArgs parse_input_args(const std::vector<std::string>& args_vec) {
    InputArgs input_args;

    if (test_args::has_command_option(args_vec, "--cabling-descriptor-path")) {
        TT_FATAL(
            test_args::has_command_option(args_vec, "--deployment-descriptor-path"),
            "Deployment Descriptor Path is required when Cabling Descriptor Path is provided.");
        input_args.cabling_descriptor_path = test_args::get_command_option(args_vec, "--cabling-descriptor-path");
    }
    if (test_args::has_command_option(args_vec, "--deployment-descriptor-path")) {
        TT_FATAL(
            input_args.cabling_descriptor_path.has_value(),
            "Cabling Descriptor Path is required when Deployment Descriptor Path is provided.");
        input_args.deployment_descriptor_path = test_args::get_command_option(args_vec, "--deployment-descriptor-path");
    }
    if (test_args::has_command_option(args_vec, "--factory-descriptor-path")) {
        TT_FATAL(
            !(input_args.cabling_descriptor_path.has_value() || input_args.deployment_descriptor_path.has_value()),
            "Pass in either Cabling Spec + Deployment Spec or just Factory System Descriptor.");
        input_args.fsd_path = test_args::get_command_option(args_vec, "--factory-descriptor-path");
    }
    if (test_args::has_command_option(args_vec, "--global-descriptor-path")) {
        input_args.gsd_path = test_args::get_command_option(args_vec, "--global-descriptor-path");
    }
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
    log_output_rank0("Generating Fabric Management Logs in " + input_args.output_path.string());

    input_args.fail_on_warning = test_args::has_command_option(args_vec, "--hard-fail");
    input_args.print_routing_tables = test_args::has_command_option(args_vec, "--print-routing-tables");
    input_args.print_ethernet_channels = test_args::has_command_option(args_vec, "--print-ethernet-channels");
    input_args.print_active_connections = test_args::has_command_option(args_vec, "--print-active-connections");
    input_args.print_all_connections = test_args::has_command_option(args_vec, "--print-all-connections");
    input_args.configure_fabric = test_args::has_command_option(args_vec, "--configure-fabric");
    input_args.initialize_fabric = test_args::has_command_option(args_vec, "--initialize-fabric");
    input_args.help = test_args::has_command_option(args_vec, "--help");

    return input_args;
}

void print_usage_info() {
    std::cout << "Utility to manage Fabric Configuration and Routing for a Multi-Node TT Cluster" << std::endl;
    std::cout << "Provides fabric management capabilities including configuration, initialization, and monitoring"
              << std::endl
              << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  --cabling-descriptor-path: Path to cabling descriptor" << std::endl;
    std::cout << "  --deployment-descriptor-path: Path to deployment descriptor" << std::endl;
    std::cout << "  --factory-descriptor-path: Path to factory descriptor" << std::endl;
    std::cout << "  --global-descriptor-path: Path to global descriptor" << std::endl;
    std::cout << "  --output-path: Path to output directory" << std::endl;
    std::cout << "  --hard-fail: Fail on warning" << std::endl;
    std::cout << "  --print-routing-tables: Print fabric routing tables" << std::endl;
    std::cout << "  --print-ethernet-channels: Print ethernet channel information" << std::endl;
    std::cout << "  --print-active-connections: Print active ethernet connections" << std::endl;
    std::cout << "  --print-all-connections: Print all ethernet connections" << std::endl;
    std::cout << "  --configure-fabric: Configure fabric routing tables" << std::endl;
    std::cout << "  --initialize-fabric: Initialize fabric configuration" << std::endl;
    std::cout << "  --fabric-config: Fabric configuration mode (FABRIC_1D, FABRIC_2D, etc.)" << std::endl;
    std::cout << "  --reliability-mode: Fabric reliability mode (STRICT_SYSTEM_HEALTH_SETUP_MODE, etc.)" << std::endl;
    std::cout << "  --num-routing-planes: Number of routing planes" << std::endl;
    std::cout << "  --fabric-tensix-config: Fabric tensix configuration (DISABLED, MUX)" << std::endl;
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

    // Configure fabric if requested
    if (input_args.configure_fabric) {
        configure_fabric_routing(
            input_args.fabric_config,
            input_args.reliability_mode,
            input_args.num_routing_planes,
            input_args.fabric_tensix_config,
            input_args.output_path);
    }

    // Print fabric information if requested
    if (*distributed_context.rank() == 0) {
        print_fabric_information(
            input_args.print_routing_tables,
            input_args.print_ethernet_channels,
            input_args.print_active_connections,
            input_args.print_all_connections,
            input_args.output_path);
    }

    distributed_context.barrier();
    return 0;
}
