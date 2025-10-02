// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <string>
#include <optional>

#include <factory_system_descriptor/utils.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt-metalium/distributed.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include "tests/tt_metal/test_utils/test_common.hpp"
#include <cabling_generator/cabling_generator.hpp>

// Captures current list of supported inputargs
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
};

// Struct to store connection information
// Used to organize and print connection information
struct ConnectionInfo {
    tt::tt_metal::AsicID asic_id;
    uint8_t channel;
    std::string host;
    tt::tt_metal::TrayID tray_id;
    tt::tt_metal::ASICLocation asic_location;
    tt::scaleout_tools::PortType port_type;
    tt::tt_metal::AsicID connected_asic_id;
    uint8_t connected_channel;
    std::string connected_host;
    tt::tt_metal::TrayID connected_tray_id;
    tt::tt_metal::ASICLocation connected_asic_location;
    tt::tt_metal::EthernetMetrics metrics;
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
    log_output_rank0("Generating System Validation Logs in " + input_args.output_path.string());

    input_args.fail_on_warning = test_args::has_command_option(args_vec, "--hard-fail");
    input_args.log_ethernet_metrics = test_args::has_command_option(args_vec, "--log-ethernet-metrics");
    input_args.print_connectivity = test_args::has_command_option(args_vec, "--print-connectivity");
    input_args.help = test_args::has_command_option(args_vec, "--help");

    TT_FATAL(
        input_args.help || input_args.cabling_descriptor_path.has_value() || input_args.fsd_path.has_value(),
        "Cluster Validation requires either Cabling Spec + Deployment Spec or a Factory System Descriptor.");

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
            context.rtoptions().get_mock_enabled(),
            run_discovery);
        log_output_rank0("Physical Discovery Complete");
        log_output_rank0("Detected Hosts: " + log_hostnames(physical_system_descriptor.get_all_hostnames()));
        return physical_system_descriptor;
    }
}

std::unordered_map<tt::tt_metal::AsicID, std::unordered_map<tt::tt_fabric::chan_id_t, tt::scaleout_tools::PortType>>
generate_port_types(const PhysicalSystemDescriptor& physical_system_descriptor) {
    std::unordered_map<tt::tt_metal::AsicID, std::unordered_map<tt::tt_fabric::chan_id_t, tt::scaleout_tools::PortType>>
        port_types;
    const auto& asic_connectivity_graph = physical_system_descriptor.get_system_graph().asic_connectivity_graph;

    for (const auto& [asic_id, asic_descriptor] : physical_system_descriptor.get_asic_descriptors()) {
        auto board_type = asic_descriptor.board_type;
        auto board = tt::scaleout_tools::create_board(board_type);
        // PhysicalSystemDescriptor internally validates that hostnames across asic descriptors are part of the graph
        // This can't throw
        const auto& asic_edges = asic_connectivity_graph.at(asic_descriptor.host_name).at(asic_id);
        for (const auto& [dst_asic_id, eth_connections] : asic_edges) {
            for (const auto& eth_connection : eth_connections) {
                auto port = board.get_port_for_asic_channel(tt::scaleout_tools::AsicChannel{
                    *(asic_descriptor.asic_location), tt::scaleout_tools::ChanId{eth_connection.src_chan}});
                port_types[asic_id][eth_connection.src_chan] = port.port_type;
            }
        }
    }
    return port_types;
}

std::string log_ethernet_metrics_and_print_connectivity(
    const InputArgs& input_args, const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) {
    std::stringstream unexpected_status_log;

    if (input_args.log_ethernet_metrics || input_args.print_connectivity) {
        log_output_rank0("Generating Ethernet Logs");
    }

    auto port_types = generate_port_types(physical_system_descriptor);

    // Collect all connections and organize by: connection_type -> hostname -> port_type -> connections
    // Using map with bool key: true = cross-host, false = local
    std::map<bool, std::map<std::string, std::map<std::string_view, std::vector<ConnectionInfo>>>>
        organized_connections;

    for (const auto& host : physical_system_descriptor.get_all_hostnames()) {
        for (const auto& [asic_id, channel_metrics] : physical_system_descriptor.get_ethernet_metrics()) {
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
                    const auto& connected_host = physical_system_descriptor.get_host_name_for_asic(connected_asic_id);
                    auto port_type_str = enchantum::to_string(port_types.at(asic_id).at(channel));

                    ConnectionInfo conn_info{
                        .asic_id = asic_id,
                        .channel = channel,
                        .host = host,
                        .tray_id = tray_id,
                        .asic_location = asic_location,
                        .port_type = port_types.at(asic_id).at(channel),
                        .connected_asic_id = connected_asic_id,
                        .connected_channel = connected_channel,
                        .connected_host = connected_host,
                        .connected_tray_id = connected_tray_id,
                        .connected_asic_location = connected_asic_location,
                        .metrics = metrics};

                    // Organize: connection_type -> hostname -> port_type -> connections
                    bool is_cross_host = (host != connected_host);
                    organized_connections[is_cross_host][host][port_type_str].push_back(conn_info);

                    if (metrics.retrain_count > 0) {
                        unexpected_status_log << " Host: " << host << " Unique ID: " << std::hex << *asic_id
                                              << " Tray: " << *tray_id << " ASIC: " << *asic_location
                                              << " Channel: " << +channel << " Retrain Count: " << metrics.retrain_count
                                              << std::endl;
                    }
                }
            }
        }
    }

    // Print organized connections: connection_type -> hostname -> port_type -> connections
    if (input_args.print_connectivity || input_args.log_ethernet_metrics) {
        // Iterate through connection types (true=cross-host, false=local)
        for (const auto& [is_cross_host, hosts_map] : organized_connections) {
            if (hosts_map.empty()) {
                continue;
            }

            // Print connection type header
            std::cout << std::endl;
            if (is_cross_host) {
                std::cout << " ============================== CROSS-HOST CONNECTIONS =============================== "
                          << std::endl;
            } else {
                std::cout << " ============================== HOST-LOCAL CONNECTIONS =============================== "
                          << std::endl;
            }

            // Iterate through hostnames
            for (const auto& [hostname, port_types_map] : hosts_map) {
                if (port_types_map.empty()) {
                    continue;
                }

                // Print hostname header
                std::cout << std::endl
                          << "  =============================== Hostname: " << hostname
                          << " =============================== " << std::endl;

                // Iterate through port types
                for (const auto& [port_type, connections] : port_types_map) {
                    if (connections.empty()) {
                        continue;
                    }

                    // Print port type header
                    std::cout << std::endl
                              << "             ---------------------- Port Type: " << port_type
                              << " ---------------------- " << std::endl
                              << std::endl;

                    // Print all connections for this port type
                    for (const auto& conn : connections) {
                        std::cout << " [" << conn.host << "] Unique ID: " << std::hex << *conn.asic_id
                                  << " Tray: " << std::dec << *conn.tray_id << ", ASIC Location: " << std::dec
                                  << *conn.asic_location << ", Ethernet Channel: " << std::dec << +conn.channel
                                  << std::endl;

                        if (input_args.print_connectivity) {
                            std::cout << "\tConnected to [" << conn.connected_host << "] Unique ID: " << std::hex
                                      << *conn.connected_asic_id << " Tray: " << std::dec << *conn.connected_tray_id
                                      << ", ASIC Location: " << std::dec << *conn.connected_asic_location
                                      << ", Ethernet Channel: " << std::dec << +conn.connected_channel << std::endl;
                        }
                        if (input_args.log_ethernet_metrics) {
                            std::cout << "\t Retrain Count: " << std::dec << conn.metrics.retrain_count << " ";
                            std::cout << "CRC Errors: 0x" << std::hex << conn.metrics.crc_error_count << " ";
                            std::cout << "Corrected Codewords: 0x" << std::hex << conn.metrics.corrected_codeword_count
                                      << " ";
                            std::cout << "Uncorrected Codewords: 0x" << std::hex
                                      << conn.metrics.uncorrected_codeword_count << " ";
                        }
                        std::cout << std::endl;
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
    std::cout << "  --log-ethernet-metrics: Log ethernet live ethernet statistics" << std::endl;
    std::cout << "  --print-connectivity: Print Ethernet Connectivity between ASICs" << std::endl;
    std::cout << "  --help: Print usage information" << std::endl << std::endl;
    std::cout << "To run on a multi-node cluster, use mpirun with a --hostfile option" << std::endl;
}

int main(int argc, char* argv[]) {
    auto input_args = parse_input_args(std::vector<std::string>(argv, argv + argc));
    if (input_args.help) {
        print_usage_info();
        return 0;
    }

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
