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

int main(int argc, char* argv[]) {
    std::optional<std::string> cabling_descriptor_path;
    std::optional<std::string> deployment_descriptor_path;
    std::optional<std::string> fsd_path;

    auto input_args = std::vector<std::string>(argv, argv + argc);

    if (test_args::has_command_option(input_args, "--cabling-descriptor-path")) {
        TT_FATAL(
            test_args::has_command_option(input_args, "--deployment-descriptor-path"),
            "Deployment Descriptor Path is required when Cabling Descriptor Path is provided.");
        cabling_descriptor_path = test_args::get_command_option(input_args, "--cabling-descriptor-path");
    }
    if (test_args::has_command_option(input_args, "--deployment-descriptor-path")) {
        TT_FATAL(
            cabling_descriptor_path.has_value(),
            "Cabling Descriptor Path is required when Deployment Descriptor Path is provided.");
        deployment_descriptor_path = test_args::get_command_option(input_args, "--deployment-descriptor-path");
    }
    if (test_args::has_command_option(input_args, "--factory-system-descriptor-path")) {
        TT_FATAL(
            !(cabling_descriptor_path.has_value() || deployment_descriptor_path.has_value()),
            "Pass in either Cabling Spec + Deployment Spec or just Factory System Descriptor.");
        fsd_path = test_args::get_command_option(input_args, "--factory-system-descriptor-path");
    }

    TT_FATAL(
        cabling_descriptor_path.has_value() || fsd_path.has_value(),
        "Cluster Validation requires either Cabling Spec + Deployment Spec or a Factory System Descriptor.");

    // bool fail_on_warning = test_args::has_command_option(input_args, "--hard-fail");
    bool log_ethernet_metrics = test_args::has_command_option(input_args, "--log-ethernet-metrics");
    bool print_connectivity = test_args::has_command_option(input_args, "--print-connectivity");

    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    if (*distributed_context.rank() == 0) {
        log_info(tt::LogDistributed, "Running Physical Discovery");
    }
    // Create physical system descriptor and discover the system
    auto physical_system_descriptor = tt::tt_metal::PhysicalSystemDescriptor();
    if (*distributed_context.rank() == 0) {
        log_info(tt::LogDistributed, "Physical Discovery Complete");
        log_info(tt::LogDistributed, "Detected Hosts: {}", physical_system_descriptor.get_all_hostnames());
    }

    // Set output path for the YAML file
    std::string gsd_yaml_path = "gsd.yaml";

    // Dump the discovered system to YAML
    physical_system_descriptor.dump_to_yaml(gsd_yaml_path);

    if (*distributed_context.rank() == 0) {
        std::string fsd_path_str;
        if (cabling_descriptor_path.has_value()) {
            log_info(tt::LogDistributed, "Creating Factory System Descriptor (Golden Representation)");
            tt::scaleout_tools::CablingGenerator cabling_generator(
                cabling_descriptor_path.value(), deployment_descriptor_path.value());
            fsd_path_str = "fsd/factory_system_descriptor_5_n300_lb.textproto";
            cabling_generator.emit_factory_system_descriptor(fsd_path_str);

        } else {
            fsd_path_str = fsd_path.value();
        }

        log_info(
            tt::LogDistributed,
            "Validating Factory System Descriptor (Golden Representation) against Global System Descriptor");
        tt::scaleout_tools::validate_fsd_against_gsd(fsd_path_str, "gsd.yaml");
        log_info(tt::LogDistributed, "Factory System Descriptor (Golden Representation) Validation Complete");

        if (log_ethernet_metrics || print_connectivity) {
            log_info(tt::LogDistributed, "Generating System Logs");
        }

        const auto& ethernet_metrics = physical_system_descriptor.get_ethernet_metrics();
        std::stringstream unexpected_status_log;

        for (const auto& host : physical_system_descriptor.get_all_hostnames()) {
            if (print_connectivity || log_ethernet_metrics) {
                std::cout << " =============================== Hostname: " << host
                          << " =============================== " << std::endl;
            }
            for (const auto& [asic_id, channel_metrics] : ethernet_metrics) {
                if (physical_system_descriptor.get_host_name_for_asic(asic_id) == host) {
                    auto tray_id = physical_system_descriptor.get_asic_descriptors().at(asic_id).tray_id;
                    auto asic_location = physical_system_descriptor.get_asic_descriptors().at(asic_id).asic_location;

                    for (const auto& [channel, metrics] : channel_metrics) {
                        auto [connected_asic_id, connected_channel] =
                            physical_system_descriptor.get_connected_asic_and_channel(asic_id, channel);
                        auto connected_tray_id =
                            physical_system_descriptor.get_asic_descriptors().at(connected_asic_id).tray_id;
                        if (print_connectivity || log_ethernet_metrics) {
                            std::cout << " Unique ID: " << std::hex << *asic_id << " Tray: " << std::dec << *tray_id
                                      << ", ASIC Location: " << std::dec << *asic_location
                                      << ", Ethernet Channel: " << std::dec << +channel << std::endl;
                        }
                        if (print_connectivity) {
                            auto connected_asic_location =
                                physical_system_descriptor.get_asic_descriptors().at(connected_asic_id).asic_location;
                            const auto& connected_host =
                                physical_system_descriptor.get_host_name_for_asic(connected_asic_id);
                            std::cout << "  Connected to " << connected_host << " Unique ID: " << std::hex
                                      << *connected_asic_id << " Tray: " << std::dec << *connected_tray_id
                                      << ", ASIC Location: " << std::dec << *connected_asic_location
                                      << ", Ethernet Channel: " << std::dec << +connected_channel << std::endl;
                        }
                        if (log_ethernet_metrics) {
                            std::cout << "\tRetrain Count: " << std::dec << metrics.retrain_count << " ";
                            std::cout << "CRC Errors: 0x" << std::hex << metrics.crc_error_count << " ";
                            std::cout << "Corrected Codewords: 0x" << std::hex << metrics.corrected_codeword_count
                                      << " ";
                            std::cout << "Uncorrected Codewords: 0x" << std::hex << metrics.uncorrected_codeword_count
                                      << " ";
                        }
                        if (print_connectivity || log_ethernet_metrics) {
                            std::cout << std::endl;
                        }

                        if (metrics.retrain_count > 0) {
                            unexpected_status_log << " Host: " << host << " Unique ID: " << std::hex << *asic_id
                                                  << " Tray: " << *tray_id << " ASIC: " << *asic_location
                                                  << " Channel: " << channel
                                                  << " Retrain Count: " << metrics.retrain_count << std::endl;
                        }
                    }
                }
            }
        }
        if (!unexpected_status_log.str().empty()) {
            std::cout << std::endl;
            std::cout << " =============================== Logging Unexpected Ethernet Statuses "
                         "=============================== "
                      << std::endl;
            std::cout << unexpected_status_log.str() << std::endl;
        }
    }
    distributed_context.barrier();
    return 0;
}
