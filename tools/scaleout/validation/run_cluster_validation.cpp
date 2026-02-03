// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <string>
#include <string_view>
#include <optional>
#include <chrono>
#include <sstream>
#include <unordered_map>

#include <cxxopts.hpp>
#include <factory_system_descriptor/utils.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt-metalium/distributed.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include <cabling_generator/cabling_generator.hpp>
#include <tt-metalium/hal.hpp>
#include "tools/scaleout/validation/utils/cluster_validation_utils.hpp"
#include <yaml-cpp/yaml.h>
#include "protobuf/factory_system_descriptor.pb.h"
#include <llrt/tt_cluster.hpp>

namespace tt::scaleout_tools {

using tt::tt_metal::AsicTopology;
using tt::tt_metal::PhysicalSystemDescriptor;

enum class CommandMode {
    VALIDATE,
    LINK_RETRAIN,
};

// Captures current list of supported input args
struct InputArgs {
    CommandMode mode = CommandMode::VALIDATE;
    std::optional<std::string> cabling_descriptor_path = std::nullopt;
    std::optional<std::string> deployment_descriptor_path = std::nullopt;
    std::optional<std::string> fsd_path = std::nullopt;
    std::optional<std::string> gsd_path = std::nullopt;
    std::filesystem::path output_path = "";
    bool fail_on_warning = false;
    bool log_ethernet_metrics = false;
    bool print_connectivity = false;
    bool help = false;
    bool send_traffic = false;
    uint32_t data_size = 0;
    uint32_t packet_size_bytes = 64;
    uint32_t num_iterations = 50;
    bool sweep_traffic_configs = false;
    bool validate_connectivity = true;
    std::optional<uint32_t> min_connections = std::nullopt;  // Relaxed validation mode

    // link_reset subcommand args
    std::optional<std::string> reset_host = std::nullopt;
    std::optional<uint32_t> reset_tray_id = std::nullopt;
    std::optional<uint32_t> reset_asic_location = std::nullopt;
    std::optional<uint32_t> reset_channel = std::nullopt;
};

std::filesystem::path generate_output_dir() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    std::string dir_name = ss.str();
    const auto& rt_options = tt::tt_metal::MetalContext::instance().rtoptions();
    std::filesystem::path output_dir_path = rt_options.get_root_dir() + "cluster_validation_logs/" + dir_name;
    std::filesystem::create_directories(output_dir_path);
    return output_dir_path;
}

cxxopts::Options create_validation_options() {
    cxxopts::Options options(
        "run_cluster_validation",
        "Utility to validate Ethernet Links and Connections for a Multi-Node TT Cluster.\n"
        "Compares live system state against the requested Cabling and Deployment Specifications.\n\n"
        "Usage:\n"
        "  run_cluster_validation [OPTIONS]                # Run validation (default)\n"
        "  run_cluster_validation link_reset [OPTIONS]     # Restart a specific cable/link\n\n"
        "The cabling-descriptor-path can be a single .textproto file or a directory containing\n"
        "multiple .textproto files that will be merged together.\n\n"
        "To run on a multi-node cluster, use mpirun with a --hostfile option.");

    options.add_options()(
        "cabling-descriptor-path",
        "Path to cabling descriptor file or directory containing multiple descriptors",
        cxxopts::value<std::string>())(
        "deployment-descriptor-path", "Path to deployment descriptor", cxxopts::value<std::string>())(
        "factory-descriptor-path", "Path to factory descriptor", cxxopts::value<std::string>())(
        "global-descriptor-path", "Path to global descriptor", cxxopts::value<std::string>())(
        "output-path", "Path to output directory", cxxopts::value<std::string>())(
        "hard-fail", "Fail on warning", cxxopts::value<bool>()->default_value("false"))(
        "log-ethernet-metrics", "Log live ethernet statistics", cxxopts::value<bool>()->default_value("false"))(
        "print-connectivity",
        "Print Ethernet Connectivity between ASICs",
        cxxopts::value<bool>()->default_value("false"))(
        "send-traffic", "Send traffic across detected links", cxxopts::value<bool>()->default_value("false"))(
        "num-iterations", "Number of iterations to send traffic", cxxopts::value<uint32_t>()->default_value("50"))(
        "data-size", "Data size (bytes) sent across each link per iteration", cxxopts::value<uint32_t>())(
        "packet-size-bytes",
        "Packet size (bytes) sent across each link",
        cxxopts::value<uint32_t>()->default_value("64"))(
        "sweep-traffic-configs",
        "Sweep pre-generated traffic configurations across detected links (stress testing)",
        cxxopts::value<bool>()->default_value("false"))(
        "min-connections",
        "Minimum connections per ASIC pair required for relaxed validation mode",
        cxxopts::value<uint32_t>())("h,help", "Print usage information");

    return options;
}

cxxopts::Options create_link_reset_options() {
    cxxopts::Options options("run_cluster_validation link_reset", "Restart a specific cable/link on the cluster.");

    options.add_options()("host", "Host name of the source ASIC", cxxopts::value<std::string>())(
        "tray-id", "Tray ID of the source ASIC", cxxopts::value<uint32_t>())(
        "asic-location", "ASIC location of the source ASIC", cxxopts::value<uint32_t>())(
        "channel", "Channel ID to reset", cxxopts::value<uint32_t>())("h,help", "Print usage information");

    return options;
}

void parse_link_reset_args(int argc, char* argv[], InputArgs& input_args) {
    input_args.mode = CommandMode::LINK_RETRAIN;
    auto options = create_link_reset_options();

    try {
        // Skip the first two args (program name and "link_reset" subcommand)
        auto result = options.parse(argc - 1, argv + 1);

        if (result.contains("help")) {
            input_args.help = true;
            return;
        }

        // Validate that all required parameters are provided
        if (result.contains("host") && result.contains("tray-id") && result.contains("asic-location") &&
            result.contains("channel")) {
            input_args.reset_host = result["host"].as<std::string>();
            input_args.reset_tray_id = result["tray-id"].as<uint32_t>();
            input_args.reset_asic_location = result["asic-location"].as<uint32_t>();
            input_args.reset_channel = result["channel"].as<uint32_t>();
        } else {
            TT_FATAL(
                false, "All link_reset parameters must be specified: --host, --tray-id, --asic-location, --channel");
        }
    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing link_reset arguments: " << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        exit(1);
    }
}

void parse_validation_args(int argc, char* argv[], InputArgs& input_args) {
    input_args.mode = CommandMode::VALIDATE;
    auto options = create_validation_options();

    try {
        auto result = options.parse(argc, argv);

        if (result.contains("help")) {
            input_args.help = true;
            return;
        }

        // Parse cabling descriptor path
        if (result.contains("cabling-descriptor-path")) {
            input_args.cabling_descriptor_path = result["cabling-descriptor-path"].as<std::string>();
        }

        // Parse deployment descriptor path
        if (result.contains("deployment-descriptor-path")) {
            TT_FATAL(
                input_args.cabling_descriptor_path.has_value(),
                "Cabling Descriptor Path is required when Deployment Descriptor Path is provided.");
            input_args.deployment_descriptor_path = result["deployment-descriptor-path"].as<std::string>();
        }

        // Parse factory descriptor path
        if (result.contains("factory-descriptor-path")) {
            TT_FATAL(
                !(input_args.cabling_descriptor_path.has_value() || input_args.deployment_descriptor_path.has_value()),
                "Pass in either Cabling Spec + Deployment Spec or just Factory System Descriptor.");
            input_args.fsd_path = result["factory-descriptor-path"].as<std::string>();
        }

        // Parse global descriptor path
        if (result.contains("global-descriptor-path")) {
            input_args.gsd_path = result["global-descriptor-path"].as<std::string>();
        }

        // Parse output path
        if (result.contains("output-path")) {
            input_args.output_path = std::filesystem::path(result["output-path"].as<std::string>());
        } else {
            input_args.output_path = generate_output_dir();
        }

        // Parse num iterations
        input_args.num_iterations = result["num-iterations"].as<uint32_t>();

        // Parse data size
        if (result.contains("data-size")) {
            input_args.data_size = result["data-size"].as<uint32_t>();
            TT_FATAL(
                input_args.data_size <= tt::tt_metal::hal::get_erisc_l1_unreserved_size(),
                "Data size must be less than or equal to the L1 unreserved size: {} bytes",
                tt::tt_metal::hal::get_erisc_l1_unreserved_size());
        } else {
            input_args.data_size = align_down(tt::tt_metal::hal::get_erisc_l1_unreserved_size(), 64);
        }

        // Parse packet size
        input_args.packet_size_bytes = result["packet-size-bytes"].as<uint32_t>();
        TT_FATAL(
            input_args.data_size % input_args.packet_size_bytes == 0, "Data size must be divisible by packet size");
        TT_FATAL(input_args.packet_size_bytes % 16 == 0, "Packet size must be divisible by 16");

        log_output_rank0("Generating System Validation Logs in " + input_args.output_path.string());

        // Parse boolean flags
        input_args.fail_on_warning = result["hard-fail"].as<bool>();
        input_args.log_ethernet_metrics = result["log-ethernet-metrics"].as<bool>();
        input_args.print_connectivity = result["print-connectivity"].as<bool>();
        input_args.send_traffic = result["send-traffic"].as<bool>();
        input_args.sweep_traffic_configs = result["sweep-traffic-configs"].as<bool>();
        input_args.validate_connectivity =
            input_args.cabling_descriptor_path.has_value() || input_args.fsd_path.has_value();

        // Parse min-connections
        if (result.contains("min-connections")) {
            uint32_t min_conn_value = result["min-connections"].as<uint32_t>();
            TT_FATAL(min_conn_value > 0, "Minimum connections must be a positive integer.");
            input_args.min_connections = min_conn_value;
            log_output_rank0(
                "Relaxed validation mode enabled. Minimum connections per ASIC pair: " +
                std::to_string(input_args.min_connections.value()));
        }

    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        exit(1);
    }
}

InputArgs parse_input_args(int argc, char* argv[]) {
    InputArgs input_args;

    // Check for top-level help first
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        input_args.help = true;
        return input_args;
    }

    // Check for subcommand and dispatch to appropriate parser
    if (argc > 1 && std::string(argv[1]) == "link_reset") {
        parse_link_reset_args(argc, argv, input_args);
    } else {
        parse_validation_args(argc, argv, input_args);
    }

    return input_args;
}

PhysicalSystemDescriptor generate_physical_system_descriptor(const InputArgs& input_args) {
    auto log_hostnames = [&](const std::vector<std::string>& hostnames) {
        std::stringstream ss;
        for (const auto& hostname : hostnames) {
            ss << hostname << ", ";
        }
        return ss.str();
    };

    if (input_args.gsd_path.has_value()) {
        auto physical_system_descriptor = tt::tt_metal::PhysicalSystemDescriptor(input_args.gsd_path.value());
        log_output_rank0("Detected Hosts: " + log_hostnames(physical_system_descriptor.get_all_hostnames()));
        return physical_system_descriptor;
    }
    log_output_rank0("Running Physical Discovery");
    constexpr bool run_discovery = true;
    auto& context = tt::tt_metal::MetalContext::instance();
    const auto& driver = context.get_cluster().get_driver();
    auto physical_system_descriptor = tt::tt_metal::PhysicalSystemDescriptor(
        driver, context.get_distributed_context_ptr(), &context.hal(), context.rtoptions(), run_discovery);
    log_output_rank0("Physical Discovery Complete");
    log_output_rank0("Detected Hosts: " + log_hostnames(physical_system_descriptor.get_all_hostnames()));
    return physical_system_descriptor;
}

AsicTopology run_connectivity_validation(
    const InputArgs& input_args, PhysicalSystemDescriptor& physical_system_descriptor) {
    if (!input_args.validate_connectivity) {
        return {};
    }
    YAML::Node gsd_yaml_node = physical_system_descriptor.generate_yaml_node();
    auto fsd_proto = get_factory_system_descriptor(
        input_args.cabling_descriptor_path,
        input_args.deployment_descriptor_path,
        input_args.fsd_path,
        physical_system_descriptor.get_all_hostnames());
    auto missing_topology = validate_connectivity(
        fsd_proto, gsd_yaml_node, input_args.fail_on_warning, physical_system_descriptor, input_args.min_connections);

    return missing_topology;
}

void print_usage_info(CommandMode mode = CommandMode::VALIDATE) {
    if (mode == CommandMode::LINK_RETRAIN) {
        auto options = create_link_reset_options();
        std::cout << options.help() << std::endl;
    } else {
        auto options = create_validation_options();
        std::cout << options.help() << std::endl;
        std::cout << "link_reset Subcommand:" << std::endl;
        std::cout << "  Use 'run_cluster_validation link_reset --help' for link_reset options." << std::endl;
    }
}

void set_config_vars() {
    // This tool must be run with slow dispatch mode, since
    // it writes custom kernels to ethernet cores, which shouldn't
    // be running fabric routers
    setenv("TT_METAL_SLOW_DISPATCH_MODE", "1", 1);

    // Only set these if they are not already set
    if (getenv("TT_MESH_HOST_RANK") == nullptr) {
        setenv("TT_MESH_HOST_RANK", "0", 1);
    }
    if (getenv("TT_MESH_ID") == nullptr) {
        setenv("TT_MESH_ID", "0", 1);
    }
    // Disable 2-ERISC mode for Blackhole
    if (getenv("TT_METAL_DISABLE_MULTI_AERISC") == nullptr) {
        setenv("TT_METAL_DISABLE_MULTI_AERISC", "1", 1);
    }
}

}  // namespace tt::scaleout_tools

int main(int argc, char* argv[]) {
    using namespace tt::scaleout_tools;

    set_config_vars();

    auto input_args = parse_input_args(argc, argv);
    if (input_args.help) {
        print_usage_info(input_args.mode);
        return 0;
    }
    bool eth_connections_healthy = true;
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();

    // Create physical system descriptor and discover the system
    auto physical_system_descriptor = generate_physical_system_descriptor(input_args);

    // Handle link_reset subcommand
    if (input_args.mode == CommandMode::LINK_RETRAIN) {
        perform_link_reset(
            input_args.reset_host.value(),
            input_args.reset_tray_id.value(),
            input_args.reset_asic_location.value(),
            input_args.reset_channel.value(),
            physical_system_descriptor);
        return 0;
    }

    AsicTopology missing_asic_topology = run_connectivity_validation(input_args, physical_system_descriptor);

    bool links_reset = false;
    // Ethernet Link Retraining through SW is currently only supported for Wormhole
    bool link_retrain_supported = tt::tt_metal::MetalContext::instance().get_cluster().arch() == tt::ARCH::WORMHOLE_B0;
    constexpr uint32_t MAX_RETRAINS_BEFORE_FAILURE =
        5;  // If links don't come up after 5 retrains, the system is in an unrecoverable state.
    uint32_t num_retrains = 0;
    while (!missing_asic_topology.empty() && link_retrain_supported && num_retrains < MAX_RETRAINS_BEFORE_FAILURE) {
        reset_ethernet_links(physical_system_descriptor, missing_asic_topology);
        links_reset = true;
        num_retrains++;
        physical_system_descriptor.run_discovery(true, true);
        missing_asic_topology = run_connectivity_validation(input_args, physical_system_descriptor);
    }

    if (num_retrains == MAX_RETRAINS_BEFORE_FAILURE && !missing_asic_topology.empty()) {
        TT_THROW("Encountered unrecoverable state. Please check the system and try again.");
        return -1;
    }
    if (links_reset) {
        log_output_rank0("Ethernet Links were Retrained. Please run the validation tool again to issue traffic.");
        return 0;
    }

    ConnectivityValidationConfig validation_config{
        .output_path = input_args.output_path,
        .cabling_descriptor_path = input_args.cabling_descriptor_path,
        .deployment_descriptor_path = input_args.deployment_descriptor_path,
        .fsd_path = input_args.fsd_path,
        .fail_on_warning = input_args.fail_on_warning};

    eth_connections_healthy = generate_link_metrics(
        physical_system_descriptor,
        input_args.num_iterations,
        input_args.log_ethernet_metrics,
        input_args.send_traffic,
        input_args.sweep_traffic_configs,
        input_args.packet_size_bytes,
        input_args.data_size,
        validation_config);

    if (*distributed_context.rank() == 0 && input_args.print_connectivity) {
        print_ethernet_connectivity(input_args.print_connectivity, physical_system_descriptor);
    }
    distributed_context.barrier();
    if (input_args.fail_on_warning && !eth_connections_healthy) {
        TT_THROW("Encountered unhealthy ethernet connections, listed above");
        return -1;
    }
    return 0;
}
