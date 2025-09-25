// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <string>

#include <factory_system_descriptor/utils.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt-metalium/distributed.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include <cabling_generator/cabling_generator.hpp>

int main(int argc, char* argv[]) {
    // Create physical system descriptor and discover the system
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    if (*distributed_context.rank() == 0) {
        log_info(tt::LogDistributed, "Running Physical Discovery");
    }
    auto physical_system_descriptor = tt::tt_metal::PhysicalSystemDescriptor();
    if (*distributed_context.rank() == 0) {
        log_info(tt::LogDistributed, "Physical Discovery Complete");
        log_info(tt::LogDistributed, "Detected Hosts: {}", physical_system_descriptor.get_all_hostnames());
    }
    // Set output path for the YAML file
    std::string gsd_yaml_path = "gsd.yaml";

    // Check if a custom output path was provided as command line argument
    if (argc > 1) {
        gsd_yaml_path = argv[1];
    }
    // Dump the discovered system to YAML
    physical_system_descriptor.dump_to_yaml(gsd_yaml_path);

    if (*distributed_context.rank() == 0) {
        log_info(tt::LogDistributed, "Creating Factory System Descriptor (Golden Representation)");
        tt::scaleout_tools::CablingGenerator cabling_generator(
            "tools/tests/scaleout/cabling_descriptors/5_n300_lb_superpod.textproto",
            "tools/tests/scaleout/deployment_descriptors/5_lb_deployment.textproto");
        cabling_generator.emit_factory_system_descriptor("fsd/factory_system_descriptor_5_n300_lb.textproto");
        log_info(
            tt::LogDistributed,
            "Validating Factory System Descriptor (Golden Representation) against Global System Descriptor");
        tt::scaleout_tools::validate_fsd_against_gsd("fsd/factory_system_descriptor_5_n300_lb.textproto", "gsd.yaml");
        log_info(tt::LogDistributed, "Factory System Descriptor (Golden Representation) Validation Complete");

        log_info(tt::LogDistributed, "Logging Ethernet Metrics");
        const auto& ethernet_metrics = physical_system_descriptor.get_ethernet_metrics();
        for (const auto& host : physical_system_descriptor.get_all_hostnames()) {
            std::cout << " =============================== Logging Ethernet Metrics for Host: " << host
                      << " =============================== " << std::endl;
            for (const auto& [asic_id, channel_metrics] : ethernet_metrics) {
                if (physical_system_descriptor.get_host_name_for_asic(asic_id) == host) {
                    auto tray_id = physical_system_descriptor.get_asic_descriptors().at(asic_id).tray_id;
                    auto asic_location = physical_system_descriptor.get_asic_descriptors().at(asic_id).asic_location;

                    for (const auto& [channel, metrics] : channel_metrics) {
                        auto [connected_asic_id, connected_channel] =
                            physical_system_descriptor.get_connected_asic_and_channel(asic_id, channel);
                        auto connected_tray_id =
                            physical_system_descriptor.get_asic_descriptors().at(connected_asic_id).tray_id;
                        auto connected_asic_location =
                            physical_system_descriptor.get_asic_descriptors().at(connected_asic_id).asic_location;
                        const auto& connected_host =
                            physical_system_descriptor.get_host_name_for_asic(connected_asic_id);
                        std::cout << " Tray: " << std::dec << *tray_id << ", ASIC Location: " << std::dec
                                  << *asic_location << ", Ethernet Channel: " << std::dec << +channel << std::endl;
                        std::cout << "  Connected to " << connected_host << " Tray: " << std::dec << *connected_tray_id
                                  << ", ASIC Location: " << std::dec << *connected_asic_location
                                  << ", Ethernet Channel: " << std::dec << +connected_channel << std::endl;
                        std::cout << "\tRetrain Count: " << std::dec << metrics.retrain_count << " ";
                        std::cout << "CRC Errors: 0x" << std::hex << metrics.crc_error_count << " ";
                        std::cout << "Corrected Codewords: 0x" << std::hex << metrics.corrected_codeword_count << " ";
                        std::cout << "Uncorrected Codewords: 0x" << std::hex << metrics.uncorrected_codeword_count
                                  << " ";
                        std::cout << std::endl;
                    }
                }
            }
        }
    }
    distributed_context.barrier();
    return 0;
}
