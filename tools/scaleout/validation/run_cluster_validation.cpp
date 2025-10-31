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
#include "tools/scaleout/validation/utils/cluster_validation_utils.hpp"

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
    bool log_ethernet_metrics = false;
    bool print_connectivity = false;
    bool help = false;
    bool send_traffic = false;
    uint32_t data_size = align_down(tt::tt_metal::hal::get_erisc_l1_unreserved_size(), 64);
    uint32_t packet_size_bytes = 64;
    uint32_t num_iterations = 50;
    bool sweep_traffic_configs = false;
    bool validate_connectivity = true;
    bool kill_links = false;
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
    if (test_args::has_command_option(args_vec, "--num-iterations")) {
        input_args.num_iterations = std::stoi(test_args::get_command_option(args_vec, "--num-iterations"));
    }
    if (test_args::has_command_option(args_vec, "--data-size")) {
        input_args.data_size = std::stoi(test_args::get_command_option(args_vec, "--data-size"));
        TT_FATAL(
            input_args.data_size <= tt::tt_metal::hal::get_erisc_l1_unreserved_size(),
            "Data size must be less than or equal to the L1 unreserved size: {} bytes",
            tt::tt_metal::hal::get_erisc_l1_unreserved_size());
    }
    if (test_args::has_command_option(args_vec, "--packet-size-bytes")) {
        input_args.packet_size_bytes = std::stoi(test_args::get_command_option(args_vec, "--packet-size-bytes"));
        TT_FATAL(
            input_args.data_size % input_args.packet_size_bytes == 0, "Data size must be divisible by packet size");
        TT_FATAL(input_args.packet_size_bytes % 16 == 0, "Packet size must be divisible by 16");
    }
    log_output_rank0("Generating System Validation Logs in " + input_args.output_path.string());

    input_args.fail_on_warning = test_args::has_command_option(args_vec, "--hard-fail");
    input_args.log_ethernet_metrics = test_args::has_command_option(args_vec, "--log-ethernet-metrics");
    input_args.print_connectivity = test_args::has_command_option(args_vec, "--print-connectivity");
    input_args.send_traffic = test_args::has_command_option(args_vec, "--send-traffic");
    input_args.sweep_traffic_configs = test_args::has_command_option(args_vec, "--sweep-traffic-configs");
    input_args.help = test_args::has_command_option(args_vec, "--help");
    input_args.validate_connectivity =
        input_args.cabling_descriptor_path.has_value() || input_args.fsd_path.has_value();
    input_args.kill_links = test_args::has_command_option(args_vec, "--kill-links");
    return input_args;
}

std::string get_factory_system_descriptor_path(const InputArgs& input_args) {
    std::string fsd_path;
    if (input_args.cabling_descriptor_path.has_value()) {
        const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
        log_output_rank0("Creating Factory System Descriptor (Golden Representation)");
        tt::scaleout_tools::CablingGenerator cabling_generator(
            input_args.cabling_descriptor_path.value(), input_args.deployment_descriptor_path.value());
        std::string filename =
            "generated_factory_system_descriptor_" + std::to_string(*distributed_context.rank()) + ".textproto";
        fsd_path = input_args.output_path / filename;
        cabling_generator.emit_factory_system_descriptor(fsd_path);
    } else {
        fsd_path = input_args.fsd_path.value();
    }
    return fsd_path;
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
    } else {
        log_output_rank0("Running Physical Discovery");
        constexpr bool run_discovery = true;
        auto& context = tt::tt_metal::MetalContext::instance();
        const auto& driver = context.get_cluster().get_driver();
        auto physical_system_descriptor = tt::tt_metal::PhysicalSystemDescriptor(
            driver,
            context.get_distributed_context_ptr(),
            &context.hal(),
            context.rtoptions(),
            run_discovery
        );
        log_output_rank0("Physical Discovery Complete");
        log_output_rank0("Detected Hosts: " + log_hostnames(physical_system_descriptor.get_all_hostnames()));
        return physical_system_descriptor;
    }
}

void cleanup_metadata(const InputArgs& input_args, const std::string& gsd_file, const std::string& fsd_file) {
    // Remove GSD file
    std::filesystem::remove(gsd_file);
    if (!input_args.fsd_path.has_value()) {
        // Remove FSD file
        std::filesystem::remove(fsd_file);
    } else {
        TT_FATAL(fsd_file == input_args.fsd_path.value(), "Internal error: Expected FSD File Paths to match");
    }
}

AsicTopology validate_connectivity(const InputArgs& input_args, PhysicalSystemDescriptor& physical_system_descriptor) {
    if (!input_args.validate_connectivity) {
        return {};
    }
    // Set output path for the YAML file
    auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    std::string gsd_yaml_filename = "global_system_descriptor_" + std::to_string(*distributed_context.rank()) + ".yaml";
    std::string gsd_yaml_path = input_args.output_path / gsd_yaml_filename;
    // Dump the discovered system to YAML
    physical_system_descriptor.dump_to_yaml(gsd_yaml_path);
    log_output_rank0("Validating Factory System Descriptor (Golden Representation) against Global System Descriptor");
    bool log_output = *distributed_context.rank() == 0;
    auto missing_physical_connections = tt::scaleout_tools::validate_fsd_against_gsd(
        get_factory_system_descriptor_path(input_args), gsd_yaml_path, true, input_args.fail_on_warning, log_output);
    log_output_rank0("Factory System Descriptor (Golden Representation) Validation Complete");
    // TODO (AS): We shouldn't need to dump files to disk for validation, once validate_fsd_against_gsd can support
    // comparing string representations of the FSD and GSD. For now, each rank dumps a file to disk, which gets deleted
    // post validation (for all ranks except rank 0).
    if (*distributed_context.rank() != 0) {
        cleanup_metadata(input_args, gsd_yaml_path, gsd_yaml_filename);
    }
    return generate_asic_topology_from_connections(missing_physical_connections, physical_system_descriptor);
}

void print_usage_info() {
    std::cout << "Utility to validate Ethernet Links and Connections for a Multi-Node TT Cluster" << std::endl;
    std::cout << "Compares live system state against the requested Cabling and Deployment Specifications" << std::endl
              << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  --cabling-descriptor-path: Path to cabling descriptor" << std::endl;
    std::cout << "  --deployment-descriptor-path: Path to deployment descriptor" << std::endl;
    std::cout << "  --factory-descriptor-path: Path to factory descriptor" << std::endl;
    std::cout << "  --global-descriptor-path: Path to global descriptor" << std::endl;
    std::cout << "  --output-path: Path to output directory" << std::endl;
    std::cout << "  --hard-fail: Fail on warning" << std::endl;
    std::cout << "  --log-ethernet-metrics: Log live ethernet statistics" << std::endl;
    std::cout << "  --print-connectivity: Print Ethernet Connectivity between ASICs" << std::endl;
    std::cout << "  --send-traffic: Send traffic across detected links" << std::endl;
    std::cout << "  --num-iterations: Number of iterations to send traffic" << std::endl;
    std::cout << "  --data-size: Data size (bytes) sent across each link per iteration" << std::endl;
    std::cout << "  --packet-size-bytes: Packet size (bytes) sent across each link" << std::endl;
    std::cout << "  --sweep-traffic-configs: Sweep pre-generated traffic configurations across detected links (stress "
                 "testing)"
              << std::endl;
    std::cout << "  --help: Print usage information" << std::endl << std::endl;
    std::cout << "To run on a multi-node cluster, use mpirun with a --hostfile option" << std::endl;
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

    bool eth_connections_healthy = true;
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();

    // Create physical system descriptor and discover the system
    auto physical_system_descriptor = generate_physical_system_descriptor(input_args);
    if (input_args.kill_links) {
        std::unordered_map<uint64_t, ChipId> asic_id_to_chip_id;
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        for (const auto& [chip_id, asic_id] : cluster.get_unique_chip_ids()) {
            asic_id_to_chip_id[asic_id] = chip_id;
        }
        std::unordered_map<uint64_t, std::set<uint8_t>> reset_cores;
        for (const auto& [asic_id, asic_connections] : physical_system_descriptor.get_asic_topology(physical_system_descriptor.my_host_name())) {
            for (const auto& [dst_asic_id, eth_connections] : asic_connections) {
                for (const auto& eth_connection : eth_connections) {
                    if (reset_cores[*asic_id].find(eth_connection.src_chan) != reset_cores[*asic_id].end()) {
                        TT_FATAL(reset_cores[*dst_asic_id].find(eth_connection.dst_chan) != reset_cores[*dst_asic_id].end(), "Expected channel {} on ASIC {} to already be reset", eth_connection.dst_chan, *dst_asic_id);
                        continue;
                    }
                    reset_cores[*asic_id].insert(eth_connection.src_chan);
                    reset_cores[*dst_asic_id].insert(eth_connection.dst_chan);
                    auto src_chan = eth_connection.src_chan;
                    // auto dst_chan = eth_connection.dst_chan;   
                    const auto& asic_descriptor = physical_system_descriptor.get_asic_descriptors().at(asic_id);
                    EthChannelIdentifier src_link = {
                        .host = physical_system_descriptor.my_host_name(),
                        .asic_id = asic_id,
                        .tray_id = asic_descriptor.tray_id,
                        .asic_location = asic_descriptor.asic_location,
                        .channel = src_chan,
                    };
                    std::cout << "Taking down link " << +src_chan << " on ASIC " << *asic_id << std::endl;
                    issue_reset_step(src_link, ResetAction::PORT_ACTION_DOWN);
                }
            }
        }
    
        physical_system_descriptor.run_discovery(true, true);
    }
    AsicTopology missing_asic_topology = validate_connectivity(input_args, physical_system_descriptor);
    missing_asic_topology = validate_connectivity(input_args, physical_system_descriptor);
    bool links_reset = false;

    constexpr uint32_t MAX_RETRAINS_BEFORE_FAILURE =
        5;  // If links don't come up after 5 retrains, the system is in an unrecoverable state.

    uint32_t num_retrains = 0;

    while (!missing_asic_topology.empty() && num_retrains < MAX_RETRAINS_BEFORE_FAILURE) {
        // reset_ethernet_links(physical_system_descriptor, missing_asic_topology);
        reset_ethernet_links_bh(missing_asic_topology, num_retrains);

        links_reset = true;
        num_retrains++;
        physical_system_descriptor.run_discovery(true, true);
        missing_asic_topology = validate_connectivity(input_args, physical_system_descriptor);
    }

    if (num_retrains == MAX_RETRAINS_BEFORE_FAILURE && !missing_asic_topology.empty()) {
        TT_THROW("Encountered unrecoverable state. Please check the system and try again.");
        return -1;
    }
    if (links_reset) {
        log_output_rank0("Ethernet Links were Retrained. Please run the validation tool again to issue traffic.");
        return 0;
    }

    eth_connections_healthy = generate_link_metrics(
        physical_system_descriptor,
        input_args.num_iterations,
        input_args.log_ethernet_metrics,
        input_args.send_traffic,
        input_args.sweep_traffic_configs,
        input_args.packet_size_bytes,
        input_args.data_size,
        input_args.output_path);

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
