// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <string>

#include <factory_system_descriptor/utils.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt-metalium/distributed.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include "tests/tt_metal/test_utils/test_common.hpp"
#include <cabling_generator/cabling_generator.hpp>

struct InputArgs {
    std::optional<std::string> cabling_descriptor_path = std::nullopt;
    std::optional<std::string> deployment_descriptor_path = std::nullopt;
    std::optional<std::string> fsd_path = std::nullopt;
    std::optional<std::string> gsd_path = std::nullopt;
    std::filesystem::path output_path = "";
    bool fail_on_warning = false;
    bool log_ethernet_metrics = false;
    bool print_connectivity = false;
};

void log_output_rank0(const std::string& message) {
    if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 0) {
        log_info(tt::LogDistributed, "{}", message);
    }
}

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
    if (test_args::has_command_option(args_vec, "--factory-system-descriptor-path")) {
        TT_FATAL(
            !(input_args.cabling_descriptor_path.has_value() || input_args.deployment_descriptor_path.has_value()),
            "Pass in either Cabling Spec + Deployment Spec or just Factory System Descriptor.");
        input_args.fsd_path = test_args::get_command_option(args_vec, "--factory-system-descriptor-path");
    }
    if (test_args::has_command_option(args_vec, "--global-system-descriptor-path")) {
        input_args.gsd_path = test_args::get_command_option(args_vec, "--global-system-descriptor-path");
    }
    if (test_args::has_command_option(args_vec, "--output-path")) {
        input_args.output_path = std::filesystem::path(test_args::get_command_option(args_vec, "--output-path"));
    } else {
        input_args.output_path = generate_output_dir();
    }
    log_output_rank0("Generating System Validation Logs in " + input_args.output_path.string());

    TT_FATAL(
        input_args.cabling_descriptor_path.has_value() || input_args.fsd_path.has_value(),
        "Cluster Validation requires either Cabling Spec + Deployment Spec or a Factory System Descriptor.");

    input_args.fail_on_warning = test_args::has_command_option(args_vec, "--hard-fail");
    input_args.log_ethernet_metrics = test_args::has_command_option(args_vec, "--log-ethernet-metrics");
    input_args.print_connectivity = test_args::has_command_option(args_vec, "--print-connectivity");

    return input_args;
}

std::string get_factory_system_descriptor_path(const InputArgs& input_args) {
    std::string fsd_path;
    if (input_args.cabling_descriptor_path.has_value()) {
        log_output_rank0("Creating Factory System Descriptor (Golden Representation)");
        tt::scaleout_tools::CablingGenerator cabling_generator(
            input_args.cabling_descriptor_path.value(), input_args.deployment_descriptor_path.value());
        fsd_path = input_args.output_path / "generated_factory_system_descriptor.textproto";
        cabling_generator.emit_factory_system_descriptor(fsd_path);

    } else {
        fsd_path = input_args.fsd_path.value();
    }
    return fsd_path;
}

PhysicalSystemDescriptor generate_physical_system_descriptor(const InputArgs& input_args) {
    PhysicalSystemDescriptor physical_system_descriptor(false);
    if (input_args.gsd_path.has_value()) {
        physical_system_descriptor = tt::tt_metal::PhysicalSystemDescriptor(input_args.gsd_path.value());
    } else {
        log_output_rank0("Running Physical Discovery");
        physical_system_descriptor = tt::tt_metal::PhysicalSystemDescriptor();
        log_output_rank0("Physical Discovery Complete");
    }
    log_output_rank0("Detected Hosts: " + [&](const std::vector<std::string>& hostnames) {
        std::stringstream ss;
        for (const auto& hostname : hostnames) {
            ss << hostname << ", ";
        }
        return ss.str();
    }(physical_system_descriptor.get_all_hostnames()));
    return physical_system_descriptor;
}

std::string log_ethernet_metrics_and_print_connectivity(
    const InputArgs& input_args, const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) {
    std::stringstream unexpected_status_log;

    if (input_args.log_ethernet_metrics || input_args.print_connectivity) {
        log_output_rank0("Generating Ethernet Metrics");
    }

    for (const auto& host : physical_system_descriptor.get_all_hostnames()) {
        if (input_args.print_connectivity || input_args.log_ethernet_metrics) {
            std::cout << std::endl
                      << " =============================== Hostname: " << host
                      << " =============================== " << std::endl;
        }
        for (const auto& [asic_id, channel_metrics] : physical_system_descriptor.get_ethernet_metrics()) {
            if (physical_system_descriptor.get_host_name_for_asic(asic_id) == host) {
                auto tray_id = physical_system_descriptor.get_asic_descriptors().at(asic_id).tray_id;
                auto asic_location = physical_system_descriptor.get_asic_descriptors().at(asic_id).asic_location;

                for (const auto& [channel, metrics] : channel_metrics) {
                    auto [connected_asic_id, connected_channel] =
                        physical_system_descriptor.get_connected_asic_and_channel(asic_id, channel);
                    auto connected_tray_id =
                        physical_system_descriptor.get_asic_descriptors().at(connected_asic_id).tray_id;
                    if (input_args.print_connectivity || input_args.log_ethernet_metrics) {
                        std::cout << " Unique ID: " << std::hex << *asic_id << " Tray: " << std::dec << *tray_id
                                  << ", ASIC Location: " << std::dec << *asic_location
                                  << ", Ethernet Channel: " << std::dec << +channel << std::endl;
                    }
                    if (input_args.print_connectivity) {
                        auto connected_asic_location =
                            physical_system_descriptor.get_asic_descriptors().at(connected_asic_id).asic_location;
                        const auto& connected_host =
                            physical_system_descriptor.get_host_name_for_asic(connected_asic_id);
                        std::cout << "  Connected to " << connected_host << " Unique ID: " << std::hex
                                  << *connected_asic_id << " Tray: " << std::dec << *connected_tray_id
                                  << ", ASIC Location: " << std::dec << *connected_asic_location
                                  << ", Ethernet Channel: " << std::dec << +connected_channel << std::endl;
                    }
                    if (input_args.log_ethernet_metrics) {
                        std::cout << "\tRetrain Count: " << std::dec << metrics.retrain_count << " ";
                        std::cout << "CRC Errors: 0x" << std::hex << metrics.crc_error_count << " ";
                        std::cout << "Corrected Codewords: 0x" << std::hex << metrics.corrected_codeword_count << " ";
                        std::cout << "Uncorrected Codewords: 0x" << std::hex << metrics.uncorrected_codeword_count
                                  << " ";
                    }
                    if (input_args.print_connectivity || input_args.log_ethernet_metrics) {
                        std::cout << std::endl;
                    }

                    if (metrics.retrain_count > 0) {
                        unexpected_status_log << " Host: " << host << " Unique ID: " << std::hex << *asic_id
                                              << " Tray: " << *tray_id << " ASIC: " << *asic_location
                                              << " Channel: " << channel << " Retrain Count: " << metrics.retrain_count
                                              << std::endl;
                    }
                }
            }
        }
    }
    return unexpected_status_log.str();
}

void log_unexpected_statuses(const std::string& unexpected_status_log) {
    if (!unexpected_status_log.empty()) {
        std::cout << std::endl;
        std::cout << " =============================== Logging Unexpected Ethernet Link Statuses "
                     "=============================== "
                  << std::endl;
        std::cout << unexpected_status_log << std::endl;
    }
}

int main(int argc, char* argv[]) {
    auto input_args = parse_input_args(std::vector<std::string>(argv, argv + argc));
    bool eth_connections_healthy = true;
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();

    // Create physical system descriptor and discover the system
    auto physical_system_descriptor = generate_physical_system_descriptor(input_args);
    // Set output path for the YAML file
    std::string gsd_yaml_path = input_args.output_path / "global_system_descriptor.yaml";
    // Dump the discovered system to YAML
    physical_system_descriptor.dump_to_yaml(gsd_yaml_path);

    if (*distributed_context.rank() == 0) {
        log_output_rank0(
            "Validating Factory System Descriptor (Golden Representation) against Global System Descriptor");
        tt::scaleout_tools::validate_fsd_against_gsd(
            get_factory_system_descriptor_path(input_args), gsd_yaml_path, true, input_args.fail_on_warning);
        log_output_rank0("Factory System Descriptor (Golden Representation) Validation Complete");

        std::string unexpected_status_log =
            log_ethernet_metrics_and_print_connectivity(input_args, physical_system_descriptor);
        log_unexpected_statuses(unexpected_status_log);
    }
    distributed_context.barrier();
    if (input_args.fail_on_warning && !eth_connections_healthy) {
        TT_THROW("Encountered unhealthy ethernet connections, listed above");
        return -1;
    }
    return 0;
}
