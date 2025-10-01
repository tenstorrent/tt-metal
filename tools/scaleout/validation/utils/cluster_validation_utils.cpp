// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/scaleout/validation/utils/cluster_validation_utils.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <thread>

#include "tests/tt_metal/test_utils/test_common.hpp"
#include "tools/scaleout/validation/utils/ethernet_link_metrics_serialization.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <enchantum/enchantum.hpp>

namespace tt::scaleout_tools {

// ============================================================================
// Data Structures
// ============================================================================

struct TrafficConfig {
    uint32_t data_size;
    uint32_t packet_size_bytes;
};

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
};

struct LinkMetricsResult {
    std::vector<EthernetLinkMetrics>
        all_link_metrics;  // All metrics for all iterations (when log_ethernet_metrics is true)
    std::vector<EthernetLinkMetrics> unhealthy_links;  // Only unhealthy links
};

// ============================================================================
// Utility Functions
// ============================================================================

void log_output_rank0(const std::string& message) {
    if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 0) {
        log_info(tt::LogDistributed, "{}", message);
    }
}

std::vector<TrafficConfig> generate_sweep_traffic_configs() {
    std::vector<TrafficConfig> configs;

    // Sweep data_size as powers of 2
    for (uint32_t data_size = 16; data_size <= 131072; data_size *= 2) {
        // For each data_size, sweep packet_size from 16 up to min(data_size, 128)
        uint32_t max_packet_size = std::min(data_size, 128u);
        for (uint32_t packet_size = 16; packet_size <= max_packet_size; packet_size *= 2) {
            configs.push_back({data_size, packet_size});
        }
    }

    return configs;
}

// ============================================================================
// Traffic Configuration Functions
// ============================================================================

void configure_local_kernels(
    const PhysicalSystemDescriptor& physical_system_descriptor,
    std::unordered_map<uint64_t, chip_id_t>& asic_id_to_chip_id,
    std::map<int, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices,
    const std::vector<uint32_t>& inputs,
    std::unordered_map<chip_id_t, tt::tt_metal::Program>& programs,
    size_t packet_size_bytes,
    size_t packet_size_words,
    size_t data_size) {
    const auto& host_name = physical_system_descriptor.my_host_name();
    const auto& asic_topology = physical_system_descriptor.get_asic_topology(host_name);
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    const size_t src_eth_l1_byte_address = tt::tt_metal::hal::get_erisc_l1_unreserved_base();
    const size_t dst_eth_l1_byte_address = tt::tt_metal::hal::get_erisc_l1_unreserved_base();

    std::unordered_map<chip_id_t, std::vector<CoreCoord>> kernel_coords;

    for (const auto& [asic_id, asic_connections] : asic_topology) {
        auto sender_chip_id = asic_id_to_chip_id[*asic_id];
        auto sender_device = devices[sender_chip_id];
        auto& sender_program = programs[sender_chip_id];

        for (const auto& [dst_asic_id, eth_connections] : asic_connections) {
            if (physical_system_descriptor.get_host_name_for_asic(dst_asic_id) != host_name) {
                continue;
            }
            auto receiver_chip_id = asic_id_to_chip_id[*dst_asic_id];
            auto receiver_device = devices[receiver_chip_id];
            auto& receiver_program = programs[receiver_chip_id];

            for (const auto& eth_connection : eth_connections) {
                auto src_chan = eth_connection.src_chan;
                auto dst_chan = eth_connection.dst_chan;

                const auto& sender_soc_desc = cluster.get_soc_desc(sender_chip_id);
                const auto& receiver_soc_desc = cluster.get_soc_desc(receiver_chip_id);
                auto sender_coord = sender_soc_desc.get_eth_core_for_channel(src_chan, CoordSystem::LOGICAL);
                auto receiver_coord = receiver_soc_desc.get_eth_core_for_channel(dst_chan, CoordSystem::LOGICAL);

                if (std::find(
                        kernel_coords[sender_chip_id].begin(), kernel_coords[sender_chip_id].end(), sender_coord) ==
                    kernel_coords[sender_chip_id].end()) {
                    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
                        sender_chip_id,
                        sender_device->ethernet_core_from_logical_core(sender_coord),
                        inputs,
                        src_eth_l1_byte_address);
                    std::vector<uint32_t> all_zeros(inputs.size(), 0);
                    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
                        receiver_chip_id,
                        receiver_device->ethernet_core_from_logical_core(receiver_coord),
                        all_zeros,
                        dst_eth_l1_byte_address);

                    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(sender_chip_id);
                    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(receiver_chip_id);

                    auto sender_kernel = tt::tt_metal::CreateKernel(
                        sender_program,
                        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_send.cpp",
                        sender_coord,
                        tt::tt_metal::EthernetConfig{
                            .noc = tt::tt_metal::NOC::NOC_0, .compile_args = {packet_size_bytes, packet_size_words}});
                    tt::tt_metal::SetRuntimeArgs(
                        sender_program,
                        sender_kernel,
                        sender_coord,
                        {src_eth_l1_byte_address, dst_eth_l1_byte_address, data_size});

                    auto receiver_kernel = tt::tt_metal::CreateKernel(
                        receiver_program,
                        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_receive.cpp",
                        receiver_coord,
                        tt::tt_metal::EthernetConfig{
                            .noc = tt::tt_metal::NOC::NOC_0,
                        });
                    tt::tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_coord, {data_size});
                    kernel_coords[sender_chip_id].push_back(sender_coord);
                    kernel_coords[receiver_chip_id].push_back(receiver_coord);
                } else {
                    TT_FATAL(
                        std::find(
                            kernel_coords[receiver_chip_id].begin(),
                            kernel_coords[receiver_chip_id].end(),
                            receiver_coord) != kernel_coords[receiver_chip_id].end(),
                        "Expected kernel to be populated for device {}, logical eth core {}",
                        receiver_chip_id,
                        receiver_coord.str());
                }
            }
        }
    }
}

void configure_cross_host_kernels(
    const PhysicalSystemDescriptor& physical_system_descriptor,
    std::unordered_map<uint64_t, chip_id_t>& asic_id_to_chip_id,
    std::map<int, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices,
    const std::vector<uint32_t>& inputs,
    std::unordered_map<chip_id_t, tt::tt_metal::Program>& programs,
    size_t packet_size_bytes,
    size_t packet_size_words,
    size_t data_size) {
    const auto& host_name = physical_system_descriptor.my_host_name();
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    const size_t src_eth_l1_byte_address = tt::tt_metal::hal::get_erisc_l1_unreserved_base();
    const size_t dst_eth_l1_byte_address = tt::tt_metal::hal::get_erisc_l1_unreserved_base();

    for (const auto& host_neighbor : physical_system_descriptor.get_host_neighbors(host_name)) {
        const auto& exit_nodes = physical_system_descriptor.get_connecting_exit_nodes(host_name, host_neighbor);
        for (const auto& exit_node : exit_nodes) {
            auto my_asic = exit_node.src_exit_node;
            auto my_chip = asic_id_to_chip_id[*my_asic];
            auto neighbor_asic = exit_node.dst_exit_node;
            bool sender = (*my_asic > *neighbor_asic);
            auto my_device = devices[my_chip];
            auto& my_program = programs[my_chip];
            const auto& my_soc_desc = cluster.get_soc_desc(my_chip);
            auto my_coord = my_soc_desc.get_eth_core_for_channel(exit_node.eth_conn.src_chan, CoordSystem::LOGICAL);

            if (sender) {
                tt::tt_metal::MetalContext::instance().get_cluster().write_core(
                    my_chip, my_device->ethernet_core_from_logical_core(my_coord), inputs, src_eth_l1_byte_address);
                tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(my_chip);
                auto sender_kernel = tt::tt_metal::CreateKernel(
                    my_program,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_send.cpp",
                    my_coord,
                    tt::tt_metal::EthernetConfig{
                        .noc = tt::tt_metal::NOC::NOC_0, .compile_args = {packet_size_bytes, packet_size_words}});
                tt::tt_metal::SetRuntimeArgs(
                    my_program, sender_kernel, my_coord, {src_eth_l1_byte_address, dst_eth_l1_byte_address, data_size});
            } else {
                std::vector<uint32_t> all_zeros(inputs.size(), 0);
                tt::tt_metal::MetalContext::instance().get_cluster().write_core(
                    my_chip, my_device->ethernet_core_from_logical_core(my_coord), all_zeros, dst_eth_l1_byte_address);
                tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(my_chip);
                auto receiver_kernel = tt::tt_metal::CreateKernel(
                    my_program,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_receive.cpp",
                    my_coord,
                    tt::tt_metal::EthernetConfig{.noc = tt::tt_metal::NOC::NOC_0});
                tt::tt_metal::SetRuntimeArgs(my_program, receiver_kernel, my_coord, {data_size});
            }
        }
    }
}

void execute_workloads(
    std::unordered_map<chip_id_t, tt::tt_metal::Program>& programs,
    std::map<int, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices) {
    std::unordered_map<chip_id_t, tt::tt_metal::distributed::MeshWorkload> mesh_workloads;

    for (auto& [device_id, program] : programs) {
        mesh_workloads[device_id] = tt::tt_metal::distributed::MeshWorkload();
        mesh_workloads[device_id].add_program(
            tt::tt_metal::distributed::MeshCoordinateRange(
                tt::tt_metal::distributed::MeshCoordinate(0, 0), tt::tt_metal::distributed::MeshCoordinate(0, 0)),
            std::move(program));
    }

    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    distributed_context.barrier();
    std::vector<std::thread> threads;
    threads.reserve(mesh_workloads.size());
    for (auto& [device_id, mesh_workload] : mesh_workloads) {
        threads.emplace_back([device_id, &mesh_workload, &devices]() {
            tt::tt_metal::distributed::EnqueueMeshWorkload(
                devices.at(device_id)->mesh_command_queue(), mesh_workload, true);
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

// ============================================================================
// Link Health Functions
// ============================================================================

bool link_unhealthy(const std::vector<LinkStatus>& link_stats) {
    auto retrain_count_increasing = [&](const std::vector<LinkStatus>& link_stats) {
        uint32_t prev_retrain_count = 0;
        for (const auto& dumped_stat : link_stats) {
            const auto& metric = dumped_stat.metrics;
            if (metric.retrain_count > prev_retrain_count) {
                return true;
            }
            prev_retrain_count = metric.retrain_count;
        }
        return false;
    };

    auto crc_error_reported = [&](const std::vector<LinkStatus>& link_stats) {
        return std::any_of(link_stats.begin(), link_stats.end(), [&](const LinkStatus& dumped_stat) {
            return dumped_stat.metrics.crc_error_count > 0;
        });
    };

    auto uncorrected_codewords_detected = [&](const std::vector<LinkStatus>& link_stats) {
        return std::any_of(link_stats.begin(), link_stats.end(), [&](const LinkStatus& dumped_stat) {
            return dumped_stat.metrics.uncorrected_codeword_count > 0;
        });
    };

    auto data_mismatch = [&](const std::vector<LinkStatus>& link_stats) {
        return std::any_of(link_stats.begin(), link_stats.end(), [&](const LinkStatus& dumped_stat) {
            return dumped_stat.num_mismatched_words > 0;
        });
    };

    return retrain_count_increasing(link_stats) || crc_error_reported(link_stats) ||
           uncorrected_codewords_detected(link_stats) || data_mismatch(link_stats);
}

LinkStatus get_first_failure(const std::vector<LinkStatus>& link_stats) {
    uint32_t prev_retrain_count = 0;

    for (const auto& link_status : link_stats) {
        // Check for retrain count increase
        if (link_status.metrics.retrain_count > prev_retrain_count) {
            return link_status;
        }

        // Check for other failures
        if (link_status.metrics.crc_error_count > 0 || link_status.metrics.uncorrected_codeword_count > 0 ||
            link_status.num_mismatched_words > 0) {
            return link_status;
        }

        prev_retrain_count = link_status.metrics.retrain_count;
    }

    // Should not reach here if link is unhealthy, but return first iteration as fallback
    if (!link_stats.empty()) {
        return link_stats[0];
    }

    return LinkStatus{};
}

// ============================================================================
// Stats and Metrics Functions
// ============================================================================

void forward_link_metrics_to_controller(std::vector<EthernetLinkMetrics>& link_metrics) {
    constexpr uint32_t CONTROLLER_RANK = 0;
    auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    auto my_rank = *distributed_context.rank();

    std::vector<uint8_t> serialized_link_metrics;
    std::size_t serialized_link_metrics_size = 0;

    if (my_rank != CONTROLLER_RANK) {
        serialized_link_metrics = tt::scaleout::validation::serialize_link_metrics_to_bytes(link_metrics);
        serialized_link_metrics_size = serialized_link_metrics.size();
        distributed_context.send(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(&serialized_link_metrics_size), sizeof(serialized_link_metrics_size)),
            tt::tt_metal::distributed::multihost::Rank{CONTROLLER_RANK},
            tt::tt_metal::distributed::multihost::Tag{0});
        distributed_context.send(
            tt::stl::as_writable_bytes(
                tt::stl::Span<uint8_t>(serialized_link_metrics.data(), serialized_link_metrics.size())),
            tt::tt_metal::distributed::multihost::Rank{CONTROLLER_RANK},
            tt::tt_metal::distributed::multihost::Tag{0});
    } else {
        for (auto peer_rank = 0; peer_rank < *distributed_context.size(); peer_rank++) {
            if (peer_rank == CONTROLLER_RANK) {
                continue;
            }
            distributed_context.recv(
                tt::stl::Span<std::byte>(
                    reinterpret_cast<std::byte*>(&serialized_link_metrics_size), sizeof(serialized_link_metrics_size)),
                tt::tt_metal::distributed::multihost::Rank{peer_rank},
                tt::tt_metal::distributed::multihost::Tag{0});
            serialized_link_metrics.resize(serialized_link_metrics_size);
            distributed_context.recv(
                tt::stl::as_writable_bytes(
                    tt::stl::Span<uint8_t>(serialized_link_metrics.data(), serialized_link_metrics.size())),
                tt::tt_metal::distributed::multihost::Rank{peer_rank},
                tt::tt_metal::distributed::multihost::Tag{0});
            std::vector<EthernetLinkMetrics> remote_link_metrics =
                tt::scaleout::validation::deserialize_link_metrics_from_bytes(serialized_link_metrics);
            link_metrics.insert(link_metrics.end(), remote_link_metrics.begin(), remote_link_metrics.end());
        }
    }
}

LinkMetricsResult process_link_statuses(
    const std::unordered_map<EthChannelIdentifier, std::vector<LinkStatus>>& statuses_per_link,
    bool log_all_ethernet_metrics) {
    LinkMetricsResult result;

    for (const auto& [channel_identifier, link_stats] : statuses_per_link) {
        bool is_unhealthy = link_unhealthy(link_stats);

        if (log_all_ethernet_metrics) {
            // Log all iterations for all links
            for (const auto& link_status : link_stats) {
                result.all_link_metrics.push_back(EthernetLinkMetrics{
                    .channel_identifier = channel_identifier,
                    .link_status = link_status,
                });
            }
        }

        if (is_unhealthy) {
            // Track unhealthy links - store the first iteration where failure occurred
            result.unhealthy_links.push_back(EthernetLinkMetrics{
                .channel_identifier = channel_identifier,
                .link_status = get_first_failure(link_stats),
            });
        }
    }

    // Forward lists to controller
    if (log_all_ethernet_metrics) {
        forward_link_metrics_to_controller(result.all_link_metrics);
    }
    forward_link_metrics_to_controller(result.unhealthy_links);

    return result;
}

void dump_link_stats(
    std::vector<uint32_t>& inputs,
    PhysicalSystemDescriptor& physical_system_descriptor,
    std::unordered_map<uint64_t, chip_id_t>& asic_id_to_chip_id,
    std::unordered_map<EthChannelIdentifier, std::vector<LinkStatus>>& statuses_per_link,
    std::map<int, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices,
    size_t data_size,
    size_t packet_size_bytes) {
    const size_t src_eth_l1_byte_address = tt::tt_metal::hal::get_erisc_l1_unreserved_base();

    const auto& host_name = physical_system_descriptor.my_host_name();
    const auto& asic_topology = physical_system_descriptor.get_asic_topology(host_name);
    const auto& asic_descriptors = physical_system_descriptor.get_asic_descriptors();
    physical_system_descriptor.generate_local_ethernet_metrics();

    for (const auto& [asic_id, asic_connections] : asic_topology) {
        auto chip_id = asic_id_to_chip_id[*asic_id];
        const auto& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(chip_id);
        for (const auto& [dst_asic_id, eth_connections] : asic_connections) {
            for (const auto& eth_connection : eth_connections) {
                auto src_chan = eth_connection.src_chan;
                auto coord = soc_desc.get_eth_core_for_channel(src_chan, CoordSystem::LOGICAL);
                uint32_t num_mismatched = 0;
                if (data_size > 0) {
                    auto result_vec = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                        chip_id,
                        devices[chip_id]->ethernet_core_from_logical_core(coord),
                        src_eth_l1_byte_address,
                        data_size);

                    // Count mismatched words
                    for (size_t i = 0; i < result_vec.size(); ++i) {
                        if (result_vec[i] != inputs[i]) {
                            num_mismatched++;
                        }
                    }
                }

                statuses_per_link[EthChannelIdentifier{
                                      .host = host_name,
                                      .asic_id = asic_descriptors.at(asic_id).unique_id,
                                      .tray_id = asic_descriptors.at(asic_id).tray_id,
                                      .asic_location = asic_descriptors.at(asic_id).asic_location,
                                      .channel = src_chan,
                                  }]
                    .push_back(LinkStatus{
                        .metrics = physical_system_descriptor.get_ethernet_metrics().at(asic_id).at(src_chan),
                        .traffic_params =
                            TrafficParams{
                                .packet_size_bytes = packet_size_bytes,
                                .data_size = data_size,
                            },
                        .num_mismatched_words = num_mismatched,
                    });
            }
        }
    }
}

// Helper function to generate random vectors
template <typename ValueType>
std::vector<ValueType> generate_uniform_random_vector(
    ValueType min, ValueType max, const size_t numel, const uint32_t seed = 0) {
    std::mt19937 gen(seed);
    std::vector<ValueType> results(numel);
    if constexpr (std::is_integral<ValueType>::value) {
        std::uniform_int_distribution<ValueType> dis(min, max);
        std::generate(results.begin(), results.end(), [&]() { return dis(gen); });
    } else if constexpr (std::is_floating_point<ValueType>::value) {
        std::uniform_real_distribution<ValueType> dis(min, max);
        std::generate(results.begin(), results.end(), [&]() { return dis(gen); });
    } else {
        std::uniform_real_distribution<float> dis(static_cast<float>(min), static_cast<float>(max));
        std::generate(results.begin(), results.end(), [&]() { return ValueType(dis(gen)); });
    }
    return results;
}

// ============================================================================
// Traffic Validation Functions
// ============================================================================

LinkMetricsResult send_traffic_and_validate_links(
    PhysicalSystemDescriptor& physical_system_descriptor,
    uint32_t num_iterations,
    bool log_ethernet_metrics,
    bool sweep_traffic_configs,
    uint32_t packet_size_bytes,
    uint32_t data_size,
    std::unordered_map<uint64_t, chip_id_t>& asic_id_to_chip_id) {
    std::vector<TrafficConfig> traffic_configs;
    if (sweep_traffic_configs) {
        traffic_configs = generate_sweep_traffic_configs();
        log_output_rank0(
            "Sweeping traffic configurations across detected links. Num Iterations: " + std::to_string(num_iterations));
    } else {
        traffic_configs = {{data_size, packet_size_bytes}};
        log_output_rank0(
            "Sending traffic across detected links. Num Iterations: " + std::to_string(num_iterations) +
            " Packet Size (Bytes): " + std::to_string(packet_size_bytes) +
            " Total Data Size: (Bytes): " + std::to_string(data_size));
    }

    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    std::vector<chip_id_t> device_ids;
    for (auto chip : cluster.all_chip_ids()) {
        device_ids.push_back(chip);
    }

    auto devices = tt::tt_metal::distributed::MeshDevice::create_unit_meshes(
        device_ids,
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        1,
        tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config());

    std::unordered_map<EthChannelIdentifier, std::vector<LinkStatus>> statuses_per_link;
    for (int i = 0; i < num_iterations; i++) {
        for (const auto& traffic_config : traffic_configs) {
            std::size_t pkt_size_bytes = traffic_config.packet_size_bytes;
            std::size_t pkt_size_words = pkt_size_bytes >> 4;
            std::size_t d_size = traffic_config.data_size;

            std::unordered_map<chip_id_t, tt::tt_metal::Program> programs;
            auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, d_size / sizeof(uint32_t));

            configure_local_kernels(
                physical_system_descriptor,
                asic_id_to_chip_id,
                devices,
                inputs,
                programs,
                pkt_size_bytes,
                pkt_size_words,
                d_size);

            configure_cross_host_kernels(
                physical_system_descriptor,
                asic_id_to_chip_id,
                devices,
                inputs,
                programs,
                pkt_size_bytes,
                pkt_size_words,
                d_size);

            execute_workloads(programs, devices);

            dump_link_stats(
                inputs,
                physical_system_descriptor,
                asic_id_to_chip_id,
                statuses_per_link,
                devices,
                d_size,
                pkt_size_bytes);
        }
    }

    return process_link_statuses(statuses_per_link, log_ethernet_metrics);
}

// ============================================================================
// Logging Functions (Metrics and Connectivity)
// ============================================================================

std::unordered_map<tt::tt_metal::AsicID, std::unordered_map<uint8_t, PortType>> generate_port_types(
    const PhysicalSystemDescriptor& physical_system_descriptor) {
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

void print_ethernet_connectivity(
    bool print_connectivity, const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) {
    if (print_connectivity) {
        log_output_rank0("Generating Ethernet Connectivity Logs");
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
                        .connected_asic_location = connected_asic_location};

                    // Organize: connection_type -> hostname -> port_type -> connections
                    bool is_cross_host = (host != connected_host);
                    organized_connections[is_cross_host][host][port_type_str].push_back(conn_info);
                }
            }
        }
    }

    // Print organized connections: connection_type -> hostname -> port_type -> connections
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

                    std::cout << "\tConnected to [" << conn.connected_host << "] Unique ID: " << std::hex
                              << *conn.connected_asic_id << " Tray: " << std::dec << *conn.connected_tray_id
                              << ", ASIC Location: " << std::dec << *conn.connected_asic_location
                              << ", Ethernet Channel: " << std::dec << +conn.connected_channel << std::endl;
                    std::cout << std::endl;
                }
            }
        }
    }
}

void log_link_metrics(
    const std::vector<EthernetLinkMetrics>& link_metrics,
    const std::filesystem::path& output_path,
    bool log_ethernet_metrics) {
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();

    if (*distributed_context.rank() != 0) {
        return;
    }

    if (link_metrics.empty()) {
        if (log_ethernet_metrics) {
            log_output_rank0("No links detected to log ethernet metrics for.");
        } else {
            log_output_rank0("✓ All Detected Links are healthy.");
        }
        return;
    }

    if (log_ethernet_metrics) {
        log_output_rank0("Generating Ethernet Metrics Report. Num Links: " + std::to_string(link_metrics.size()));
    } else {
        log_output_rank0(
            "✗ Found Unhealthy Links. Generating Failure Report (Only Pinging Detected Links). Num Unhealthy Links: " +
            std::to_string(link_metrics.size()));
    }

    struct MetricRow {
        EthChannelIdentifier channel_id;
        std::string metric_type;  // Only used when logging failures
        TrafficParams traffic_params;
        uint32_t retrain_count;
        uint32_t crc_error_count;
        uint32_t uncorrected_codeword_count;
        uint32_t num_mismatched_words;
    };

    std::vector<MetricRow> metric_rows;

    if (log_ethernet_metrics) {
        // When logging all metrics: one row per link with all metrics
        for (const auto& link : link_metrics) {
            metric_rows.push_back(
                {link.channel_identifier,
                 "",  // metric_type not used
                 link.link_status.traffic_params,
                 link.link_status.metrics.retrain_count,
                 link.link_status.metrics.crc_error_count,
                 link.link_status.metrics.uncorrected_codeword_count,
                 link.link_status.num_mismatched_words});
        }
    } else {
        // When logging failures: one row per metric type with non-zero values
        for (const auto& link : link_metrics) {
            if (link.link_status.metrics.retrain_count > 0) {
                metric_rows.push_back(
                    {link.channel_identifier,
                     "Retrain",
                     link.link_status.traffic_params,
                     link.link_status.metrics.retrain_count,
                     0,
                     0,
                     0});
            }
            if (link.link_status.metrics.crc_error_count > 0) {
                metric_rows.push_back(
                    {link.channel_identifier,
                     "CRC Error",
                     link.link_status.traffic_params,
                     0,
                     link.link_status.metrics.crc_error_count,
                     0,
                     0});
            }
            if (link.link_status.metrics.uncorrected_codeword_count > 0) {
                metric_rows.push_back(
                    {link.channel_identifier,
                     "Uncorrected CW",
                     link.link_status.traffic_params,
                     0,
                     0,
                     link.link_status.metrics.uncorrected_codeword_count,
                     0});
            }
            if (link.link_status.num_mismatched_words > 0) {
                metric_rows.push_back(
                    {link.channel_identifier,
                     "Data Mismatch",
                     link.link_status.traffic_params,
                     0,
                     0,
                     0,
                     link.link_status.num_mismatched_words});
            }
        }
    }

    // Print console table
    std::cout << std::endl;
    std::cout << "╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗"
              << std::endl;
    if (log_ethernet_metrics) {
        std::cout
            << "║                          ETHERNET METRICS REPORT                                                  ║"
            << std::endl;
    } else {
        std::cout
            << "║                              FAULTY LINKS REPORT                                                  ║"
            << std::endl;
    }
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝"
              << std::endl;
    if (log_ethernet_metrics) {
        std::cout << "Total Links: " << link_metrics.size() << std::endl;
        std::cout << "Total Metric Entries: " << metric_rows.size() << std::endl << std::endl;
    } else {
        std::cout << "Total Faulty Link Occurrences: " << link_metrics.size() << std::endl;
        std::cout << "Total Failure Instances: " << metric_rows.size() << std::endl << std::endl;
    }

    // Table header
    std::cout << std::left << std::setw(20) << "Host" << std::setw(6) << "Tray" << std::setw(6) << "ASIC"
              << std::setw(5) << "Ch" << std::setw(14) << "Unique ID" << std::setw(12) << "Retrains" << std::setw(14)
              << "CRC Err" << std::setw(18) << "Uncorrected CW" << std::setw(16) << "Mismatch Words";

    if (!log_ethernet_metrics) {
        std::cout << std::setw(18) << "Failure Type";
    }

    std::cout << std::setw(12) << "Pkt Size" << std::setw(12) << "Data Size" << std::endl;

    std::cout << std::string(log_ethernet_metrics ? 135 : 153, '-') << std::endl;

    // Table rows
    for (const auto& row : metric_rows) {
        std::cout << std::left << std::setw(20) << row.channel_id.host << std::setw(6) << *row.channel_id.tray_id
                  << std::setw(6) << *row.channel_id.asic_location << std::setw(5)
                  << static_cast<int>(row.channel_id.channel);

        // Print Unique ID in hex
        std::stringstream uid_stream;
        uid_stream << "0x" << std::hex << std::setfill('0') << std::setw(10) << *row.channel_id.asic_id;
        std::cout << std::left << std::setw(14) << uid_stream.str();

        // Retrains
        std::cout << std::dec << std::setfill(' ') << std::left << std::setw(12) << row.retrain_count;

        // CRC errors
        std::stringstream crc_stream;
        crc_stream << "0x" << std::hex << row.crc_error_count;
        std::cout << std::left << std::setw(14) << crc_stream.str();

        // Uncorrected codewords
        std::stringstream uncorr_stream;
        uncorr_stream << "0x" << std::hex << row.uncorrected_codeword_count;
        std::cout << std::left << std::setw(18) << uncorr_stream.str();

        // Mismatched words
        std::cout << std::dec << std::left << std::setw(16) << row.num_mismatched_words;

        // Failure Type (only for faulty links report)
        if (!log_ethernet_metrics) {
            std::cout << std::left << std::setw(18) << row.metric_type;
        }

        std::cout << std::setw(12) << (std::to_string(row.traffic_params.packet_size_bytes) + " B") << std::setw(12)
                  << (std::to_string(row.traffic_params.data_size) + " B") << std::endl;
    }

    std::cout << std::string(log_ethernet_metrics ? 135 : 153, '-') << std::endl << std::endl;

    // Write CSV file
    std::filesystem::path csv_path =
        log_ethernet_metrics ? output_path / "ethernet_metrics_report.csv" : output_path / "unhealthy_links_report.csv";
    std::ofstream csv_file(csv_path);

    if (csv_file.is_open()) {
        // CSV header
        csv_file << "Host,Tray,ASIC,Channel,Unique_ID";
        if (!log_ethernet_metrics) {
            csv_file << ",Failure_Type";
        }
        csv_file << ",Packet_Size_Bytes,Data_Size_Bytes,"
                 << "Retrain_Count,CRC_Error_Count,Uncorrected_Codeword_Count,Mismatched_Words" << std::endl;

        // CSV rows
        for (const auto& row : metric_rows) {
            csv_file << row.channel_id.host << "," << *row.channel_id.tray_id << "," << *row.channel_id.asic_location
                     << "," << static_cast<int>(row.channel_id.channel) << ","
                     << "0x" << std::hex << *row.channel_id.asic_id << std::dec;
            if (!log_ethernet_metrics) {
                csv_file << "," << row.metric_type;
            }
            csv_file << "," << row.traffic_params.packet_size_bytes << "," << row.traffic_params.data_size << ","
                     << row.retrain_count << ","
                     << "0x" << std::hex << row.crc_error_count << std::dec << ","
                     << "0x" << std::hex << row.uncorrected_codeword_count << std::dec << ","
                     << row.num_mismatched_words << std::endl;
        }

        csv_file.close();
        log_output_rank0("✓ Detailed report written to: " + csv_path.string());
    } else {
        log_output_rank0("✗ Warning: Could not open CSV file for writing: " + csv_path.string());
    }
}

// ============================================================================
// Link Metrics Generation
// ============================================================================

bool generate_link_metrics(
    PhysicalSystemDescriptor& physical_system_descriptor,
    uint32_t num_iterations,
    bool log_ethernet_metrics,
    bool send_traffic,
    bool sweep_traffic_configs,
    uint32_t packet_size_bytes,
    uint32_t data_size,
    const std::filesystem::path& output_path) {
    std::unordered_map<uint64_t, chip_id_t> asic_id_to_chip_id;
    for (const auto& [chip_id, asic_id] : tt::tt_metal::MetalContext::instance().get_cluster().get_unique_chip_ids()) {
        asic_id_to_chip_id[asic_id] = chip_id;
    }

    LinkMetricsResult result;

    if (send_traffic) {
        result = send_traffic_and_validate_links(
            physical_system_descriptor,
            num_iterations,
            log_ethernet_metrics,
            sweep_traffic_configs,
            packet_size_bytes,
            data_size,
            asic_id_to_chip_id);
    } else {
        std::unordered_map<EthChannelIdentifier, std::vector<LinkStatus>> statuses_per_link;
        std::vector<uint32_t> inputs = {};
        std::map<int, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> empty_devices;
        dump_link_stats(inputs, physical_system_descriptor, asic_id_to_chip_id, statuses_per_link, empty_devices, 0, 0);
        result = process_link_statuses(statuses_per_link, log_ethernet_metrics);
    }

    // Log metrics
    if (log_ethernet_metrics) {
        // Log all ethernet metrics
        log_link_metrics(result.all_link_metrics, output_path, true);
    }
    log_link_metrics(result.unhealthy_links, output_path, false);
    return result.unhealthy_links.empty();
}

}  // namespace tt::scaleout_tools
