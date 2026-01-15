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
#include <future>
#include <chrono>

#include "tools/scaleout/validation/utils/ethernet_link_metrics_serialization.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <enchantum/enchantum.hpp>
#include <cabling_generator/cabling_generator.hpp>
#include <google/protobuf/text_format.h>
#include <yaml-cpp/yaml.h>
#include "protobuf/factory_system_descriptor.pb.h"
#include <llrt/tt_cluster.hpp>

namespace tt::scaleout_tools {

// Timeout is based on the assumption that WORKLOAD_TIMEOUT_DURATION (30s) should be enough for an iteration to run.
// Any longer and we assume that a hang has been encountered. We may need to tune this in future.
static constexpr std::chrono::seconds WORKLOAD_TIMEOUT_DURATION{30};

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
    tt::scaleout_tools::PortId port_id;
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

enum class WorkloadResult { Completed, TimedOut };

struct ClusterContext {
    PhysicalSystemDescriptor& physical_system_descriptor;
    std::unordered_map<uint64_t, ChipId>& asic_id_to_chip_id;
    std::map<int, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices;
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
    ClusterContext& ctx,
    const std::vector<uint32_t>& inputs,
    std::unordered_map<ChipId, tt::tt_metal::Program>& programs,
    uint32_t packet_size_bytes,
    uint32_t packet_size_words,
    uint32_t data_size,
    bool fwd) {
    const auto& host_name = ctx.physical_system_descriptor.my_host_name();
    const auto& asic_topology = ctx.physical_system_descriptor.get_asic_topology(host_name);
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    const uint32_t src_eth_l1_byte_address = tt::tt_metal::hal::get_erisc_l1_unreserved_base();
    const uint32_t dst_eth_l1_byte_address = tt::tt_metal::hal::get_erisc_l1_unreserved_base();

    std::unordered_map<ChipId, std::vector<CoreCoord>> kernel_coords;
    std::vector<uint32_t> all_zeros(inputs.size(), 0);

    // Single ERISC Execution for BH for now, since ERISC0 needs to manage link recovery
    auto erisc_id = tt::tt_metal::DataMovementProcessor::RISCV_0;
    auto noc_id = tt::tt_metal::MetalContext::instance().get_cluster().arch() == ARCH::BLACKHOLE
                      ? tt::tt_metal::NOC::NOC_1
                      : tt::tt_metal::NOC::NOC_0;

    for (const auto& [asic_id, asic_connections] : asic_topology) {
        auto curr_chip_id = ctx.asic_id_to_chip_id[*asic_id];
        auto curr_chip = ctx.devices[curr_chip_id];
        auto& curr_program = programs[curr_chip_id];

        for (const auto& [neighbor_asic_id, eth_connections] : asic_connections) {
            if (ctx.physical_system_descriptor.get_host_name_for_asic(neighbor_asic_id) != host_name) {
                continue;
            }
            auto neighbor_chip_id = ctx.asic_id_to_chip_id[*neighbor_asic_id];
            auto neighbor_chip = ctx.devices[neighbor_chip_id];
            auto& neighbor_program = programs[neighbor_chip_id];

            for (const auto& eth_connection : eth_connections) {
                auto curr_chan = eth_connection.src_chan;
                auto neighbor_chan = eth_connection.dst_chan;

                const auto& curr_soc_desc = cluster.get_soc_desc(curr_chip_id);
                const auto& neighbor_soc_desc = cluster.get_soc_desc(neighbor_chip_id);
                auto curr_coord = curr_soc_desc.get_eth_core_for_channel(curr_chan, CoordSystem::LOGICAL);
                auto neighbor_coord = neighbor_soc_desc.get_eth_core_for_channel(neighbor_chan, CoordSystem::LOGICAL);

                if (std::find(kernel_coords[curr_chip_id].begin(), kernel_coords[curr_chip_id].end(), curr_coord) ==
                    kernel_coords[curr_chip_id].end()) {
                    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
                        curr_chip_id,
                        curr_chip->ethernet_core_from_logical_core(curr_coord),
                        fwd ? inputs : all_zeros,
                        src_eth_l1_byte_address);

                    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
                        neighbor_chip_id,
                        neighbor_chip->ethernet_core_from_logical_core(neighbor_coord),
                        fwd ? all_zeros : inputs,
                        dst_eth_l1_byte_address);

                    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(curr_chip_id);
                    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(neighbor_chip_id);

                    const auto* sender_kernel_path =
                        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_send.cpp";
                    const auto* receiver_kernel_path =
                        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_receive.cpp";
                    std::vector<uint32_t> sender_compile_args = {packet_size_bytes, packet_size_words};
                    std::vector<uint32_t> receiver_compile_args = {};
                    auto curr_kernel = tt::tt_metal::CreateKernel(
                        curr_program,
                        fwd ? sender_kernel_path : receiver_kernel_path,
                        curr_coord,
                        tt::tt_metal::EthernetConfig{
                            .noc = noc_id,
                            .processor = erisc_id,
                            .compile_args = fwd ? std::vector<uint32_t>{packet_size_bytes, packet_size_words}
                                                : std::vector<uint32_t>{}});

                    auto neighbor_kernel = tt::tt_metal::CreateKernel(
                        neighbor_program,
                        fwd ? receiver_kernel_path : sender_kernel_path,
                        neighbor_coord,
                        tt::tt_metal::EthernetConfig{
                            .noc = noc_id,
                            .processor = erisc_id,
                            .compile_args = fwd ? std::vector<uint32_t>{}
                                                : std::vector<uint32_t>{packet_size_bytes, packet_size_words}});
                    tt::tt_metal::SetRuntimeArgs(
                        fwd ? curr_program : neighbor_program,
                        fwd ? curr_kernel : neighbor_kernel,
                        fwd ? curr_coord : neighbor_coord,
                        {src_eth_l1_byte_address, dst_eth_l1_byte_address, data_size});

                    tt::tt_metal::SetRuntimeArgs(
                        fwd ? neighbor_program : curr_program,
                        fwd ? neighbor_kernel : curr_kernel,
                        fwd ? neighbor_coord : curr_coord,
                        {data_size});

                    kernel_coords[curr_chip_id].push_back(curr_coord);
                    kernel_coords[neighbor_chip_id].push_back(neighbor_coord);
                } else {
                    TT_FATAL(
                        std::find(
                            kernel_coords[neighbor_chip_id].begin(),
                            kernel_coords[neighbor_chip_id].end(),
                            neighbor_coord) != kernel_coords[neighbor_chip_id].end(),
                        "Expected kernel to be populated for device {}, logical eth core {}",
                        neighbor_chip_id,
                        neighbor_coord.str());
                }
            }
        }
    }
}

void configure_cross_host_kernels(
    ClusterContext& ctx,
    const std::vector<uint32_t>& inputs,
    std::unordered_map<ChipId, tt::tt_metal::Program>& programs,
    uint32_t packet_size_bytes,
    uint32_t packet_size_words,
    uint32_t data_size,
    bool fwd) {
    const auto& host_name = ctx.physical_system_descriptor.my_host_name();
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    const uint32_t src_eth_l1_byte_address = tt::tt_metal::hal::get_erisc_l1_unreserved_base();
    const uint32_t dst_eth_l1_byte_address = tt::tt_metal::hal::get_erisc_l1_unreserved_base();

    // Single ERISC Execution for BH for now, since ERISC0 needs to manage link recovery
    auto erisc_id = tt::tt_metal::DataMovementProcessor::RISCV_0;
    auto noc_id = tt::tt_metal::MetalContext::instance().get_cluster().arch() == ARCH::BLACKHOLE
                      ? tt::tt_metal::NOC::NOC_1
                      : tt::tt_metal::NOC::NOC_0;

    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    for (const auto& host_neighbor : ctx.physical_system_descriptor.get_host_neighbors(host_name)) {
        const auto& exit_nodes = ctx.physical_system_descriptor.get_connecting_exit_nodes(host_name, host_neighbor);
        for (const auto& exit_node : exit_nodes) {
            auto my_asic = exit_node.src_exit_node;
            auto my_chip = ctx.asic_id_to_chip_id[*my_asic];
            auto neighbor_asic = exit_node.dst_exit_node;
            bool sender = fwd ? (*my_asic > *neighbor_asic) : (*my_asic < *neighbor_asic);
            auto my_device = ctx.devices[my_chip];
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
                        .noc = noc_id, .processor = erisc_id, .compile_args = {packet_size_bytes, packet_size_words}});
                tt::tt_metal::SetRuntimeArgs(
                    my_program, sender_kernel, my_coord, {src_eth_l1_byte_address, dst_eth_l1_byte_address, data_size});
            } else {
                tt::tt_metal::MetalContext::instance().get_cluster().write_core(
                    my_chip, my_device->ethernet_core_from_logical_core(my_coord), all_zeros, dst_eth_l1_byte_address);
                tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(my_chip);
                auto receiver_kernel = tt::tt_metal::CreateKernel(
                    my_program,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_l1_direct_receive.cpp",
                    my_coord,
                    tt::tt_metal::EthernetConfig{.noc = noc_id, .processor = erisc_id});
                tt::tt_metal::SetRuntimeArgs(my_program, receiver_kernel, my_coord, {data_size});
            }
        }
    }
}

WorkloadResult execute_workloads(
    std::unordered_map<ChipId, tt::tt_metal::Program>& programs,
    std::map<int, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>>& devices) {
    std::unordered_map<ChipId, tt::tt_metal::distributed::MeshWorkload> mesh_workloads;

    for (auto& [device_id, program] : programs) {
        mesh_workloads[device_id] = tt::tt_metal::distributed::MeshWorkload();
        mesh_workloads[device_id].add_program(
            tt::tt_metal::distributed::MeshCoordinateRange(
                tt::tt_metal::distributed::MeshCoordinate(0, 0), tt::tt_metal::distributed::MeshCoordinate(0, 0)),
            std::move(program));
    }

    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    distributed_context.barrier();

    // Launch async tasks
    static std::vector<std::future<void>> futures;  // Static to avoid destructor blocking
    futures.clear();
    futures.reserve(mesh_workloads.size());
    for (auto& [device_id, mesh_workload] : mesh_workloads) {
        futures.push_back(std::async(std::launch::async, [device_id, &mesh_workload, &devices]() {
            tt::tt_metal::distributed::EnqueueMeshWorkload(
                devices.at(device_id)->mesh_command_queue(), mesh_workload, true);
        }));
    }

    // All futures are expected to complete within the timeout duration
    auto start_time = std::chrono::steady_clock::now();
    for (auto& future : futures) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        auto remaining = std::max(
            WORKLOAD_TIMEOUT_DURATION - std::chrono::duration_cast<std::chrono::seconds>(elapsed),
            std::chrono::seconds(0));

        if (future.wait_for(remaining) != std::future_status::ready) {
            // Don't wait for futures to complete because they're stuck. Just abandon them.
            // Static storage prevents destructor from blocking.
            return WorkloadResult::TimedOut;
        }
        // Get the result to propagate any exceptions
        future.get();
    }

    return WorkloadResult::Completed;
}

// ============================================================================
// Link Health Functions
// ============================================================================

bool link_unhealthy(const std::vector<LinkStatus>& link_stats) {
    auto retrain_count_increasing = [&](const std::vector<LinkStatus>& link_stats) {
        uint32_t prev_retrain_count = link_stats[0].metrics.retrain_count;
        for (const auto& dumped_stat : link_stats) {
            const auto& metric = dumped_stat.metrics;
            if (metric.retrain_count > prev_retrain_count) {
                return true;
            }
            prev_retrain_count = metric.retrain_count;
        }
        return false;
    };

    auto zero_retrain_count = [&](const std::vector<LinkStatus>& link_stats) {
        return std::any_of(link_stats.begin(), link_stats.end(), [&](const LinkStatus& dumped_stat) {
            return dumped_stat.metrics.retrain_count == 0;
        });
    };

    auto crc_error_reported = [&](const std::vector<LinkStatus>& link_stats) {
        return std::any_of(link_stats.begin(), link_stats.end(), [&](const LinkStatus& dumped_stat) {
            return dumped_stat.metrics.crc_error_count > 0;
        });
    };

    auto uncorrected_codewords_increasing = [&](const std::vector<LinkStatus>& link_stats) {
        uint32_t prev_uncorrected_codeword_count = link_stats[0].metrics.uncorrected_codeword_count;
        for (const auto& dumped_stat : link_stats) {
            const auto& metric = dumped_stat.metrics;
            if (metric.uncorrected_codeword_count > prev_uncorrected_codeword_count) {
                return true;
            }
            prev_uncorrected_codeword_count = metric.uncorrected_codeword_count;
        }
        return false;
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
    bool retrain_count_increasing_ = retrain_count_increasing(link_stats);
    bool crc_error_reported_ = crc_error_reported(link_stats);
    bool uncorrected_codewords_detected_ = uncorrected_codewords_detected(link_stats);
    bool uncorrected_codewords_increasing_ = uncorrected_codewords_increasing(link_stats);
    bool data_mismatch_ = data_mismatch(link_stats);
    bool zero_retrain_count_ = zero_retrain_count(link_stats);

    // A link is considered unhealthy if:
    // - The retrain count is increasing
    // - A CRC error is reported
    // - Uncorrected codewords are detected but no retrains were issued
    // - Uncorrected codewords are increasing
    // - A data mismatch is detected
    auto arch = tt::tt_metal::MetalContext::instance().get_cluster().arch();
    if (arch == tt::ARCH::BLACKHOLE) {
        // BH systems support real-time link retraining/recovery. As such its considered normal for link retrain
        // counts to be increasing.
        // CRC Errors are not considered a valid metric for BH systems.
        // Link health for BH is solely based on data mismatches.
        return data_mismatch_;
    }
    // For WH systems, use the extensive link health check.
    return retrain_count_increasing_ || crc_error_reported_ ||
           (zero_retrain_count_ && uncorrected_codewords_detected_) || uncorrected_codewords_increasing_ ||
           data_mismatch_;
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
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
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

struct PortInfo {
    tt::scaleout_tools::PortType port_type = tt::scaleout_tools::PortType::TRACE;
    tt::scaleout_tools::PortId port_id{0};
};

std::unordered_map<tt::tt_metal::AsicID, std::unordered_map<uint8_t, PortInfo>> generate_port_info(
    const PhysicalSystemDescriptor& physical_system_descriptor) {
    std::unordered_map<tt::tt_metal::AsicID, std::unordered_map<tt::tt_fabric::chan_id_t, PortInfo>> port_info_map;
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
                port_info_map[asic_id][eth_connection.src_chan] = PortInfo{port.port_type, port.port_id};
            }
        }
    }
    return port_info_map;
}

void dump_link_stats(
    ClusterContext& ctx,
    std::vector<uint32_t>& inputs,
    std::unordered_map<EthChannelIdentifier, std::vector<LinkStatus>>& statuses_per_link,
    uint32_t data_size,
    uint32_t packet_size_bytes) {
    const uint32_t src_eth_l1_byte_address = tt::tt_metal::hal::get_erisc_l1_unreserved_base();

    const auto& host_name = ctx.physical_system_descriptor.my_host_name();
    const auto& asic_topology = ctx.physical_system_descriptor.get_asic_topology(host_name);
    const auto& asic_descriptors = ctx.physical_system_descriptor.get_asic_descriptors();
    auto local_ethernet_metrics = ctx.physical_system_descriptor.query_local_ethernet_metrics();
    auto port_info_map = generate_port_info(ctx.physical_system_descriptor);

    for (const auto& [asic_id, asic_connections] : asic_topology) {
        auto chip_id = ctx.asic_id_to_chip_id[*asic_id];
        const auto& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(chip_id);
        for (const auto& [dst_asic_id, eth_connections] : asic_connections) {
            for (const auto& eth_connection : eth_connections) {
                auto src_chan = eth_connection.src_chan;
                auto coord = soc_desc.get_eth_core_for_channel(src_chan, CoordSystem::LOGICAL);
                uint32_t num_mismatched = 0;
                if (data_size > 0) {
                    auto result_vec = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                        chip_id,
                        ctx.devices[chip_id]->ethernet_core_from_logical_core(coord),
                        src_eth_l1_byte_address,
                        data_size);

                    // Count mismatched words
                    for (size_t i = 0; i < result_vec.size(); ++i) {
                        if (result_vec[i] != inputs[i]) {
                            num_mismatched++;
                        }
                    }
                }

                const auto& port_info = port_info_map.at(asic_id).at(src_chan);
                statuses_per_link[EthChannelIdentifier{
                                      .host = host_name,
                                      .asic_id = asic_descriptors.at(asic_id).unique_id,
                                      .tray_id = asic_descriptors.at(asic_id).tray_id,
                                      .asic_location = asic_descriptors.at(asic_id).asic_location,
                                      .channel = src_chan,
                                      .port_id = *port_info.port_id,
                                      .port_type = static_cast<uint32_t>(port_info.port_type),
                                  }]
                    .push_back(LinkStatus{
                        .metrics = local_ethernet_metrics.at(asic_id).at(src_chan),
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
    if constexpr (std::is_integral_v<ValueType>) {
        std::uniform_int_distribution<ValueType> dis(min, max);
        std::generate(results.begin(), results.end(), [&]() { return dis(gen); });
    } else if constexpr (std::is_floating_point_v<ValueType>) {
        std::uniform_real_distribution<ValueType> dis(min, max);
        std::generate(results.begin(), results.end(), [&]() { return dis(gen); });
    } else {
        std::uniform_real_distribution<float> dis(static_cast<float>(min), static_cast<float>(max));
        std::generate(results.begin(), results.end(), [&]() { return ValueType(dis(gen)); });
    }
    return results;
}

// ============================================================================
// Logging Functions (Metrics and Connectivity)
// ============================================================================

void print_ethernet_connectivity(
    bool /*print_connectivity*/, const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) {
    auto port_info_map = generate_port_info(physical_system_descriptor);

    // Collect all connections and organize by: connection_type -> hostname -> port_type -> connections
    // Using map with bool key: true = cross-host, false = local
    std::map<bool, std::map<std::string, std::map<std::string_view, std::vector<ConnectionInfo>>>>
        organized_connections;

    for (const auto& host : physical_system_descriptor.get_all_hostnames()) {
        const auto& asic_connections = physical_system_descriptor.get_asic_topology(host);
        for (auto asic_id : physical_system_descriptor.get_asics_connected_to_host(host)) {
            auto tray_id = physical_system_descriptor.get_asic_descriptors().at(asic_id).tray_id;
            auto asic_location = physical_system_descriptor.get_asic_descriptors().at(asic_id).asic_location;

            for (const auto& asic_connection : asic_connections.at(asic_id)) {
                auto connected_asic_id = asic_connection.first;
                auto connected_tray_id =
                    physical_system_descriptor.get_asic_descriptors().at(connected_asic_id).tray_id;
                auto connected_asic_location =
                    physical_system_descriptor.get_asic_descriptors().at(connected_asic_id).asic_location;
                const auto& connected_host = physical_system_descriptor.get_host_name_for_asic(connected_asic_id);

                for (const auto& eth_connection : asic_connection.second) {
                    auto channel = eth_connection.src_chan;
                    auto connected_channel = eth_connection.dst_chan;
                    const auto& port_info = port_info_map.at(asic_id).at(channel);
                    auto port_type_str = enchantum::to_string(port_info.port_type);

                    ConnectionInfo conn_info{
                        .asic_id = asic_id,
                        .channel = channel,
                        .host = host,
                        .tray_id = tray_id,
                        .asic_location = asic_location,
                        .port_type = port_info.port_type,
                        .port_id = port_info.port_id,
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
                              << ", Port ID: " << std::dec << *conn.port_id << std::endl;

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
        uint64_t corrected_codeword_count;
        uint64_t uncorrected_codeword_count;
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
                 link.link_status.metrics.corrected_codeword_count,
                 link.link_status.metrics.uncorrected_codeword_count,
                 link.link_status.num_mismatched_words});
        }
    } else {
        // When logging failures: one row per link with all metrics and combined failure types
        for (const auto& link : link_metrics) {
            std::vector<std::string> failure_types;

            if (link.link_status.metrics.retrain_count > 0) {
                failure_types.push_back("Retrain");
            }
            if (link.link_status.metrics.crc_error_count > 0) {
                failure_types.push_back("CRC Error");
            }
            if (link.link_status.metrics.uncorrected_codeword_count > 0) {
                failure_types.push_back("Uncorrected CW");
            }
            if (link.link_status.num_mismatched_words > 0) {
                failure_types.push_back("Data Mismatch");
            }

            // Combine failure types with "+"
            std::string combined_failure_type;
            for (size_t i = 0; i < failure_types.size(); ++i) {
                if (i > 0) {
                    combined_failure_type += " + ";
                }
                combined_failure_type += failure_types[i];
            }

            metric_rows.push_back(
                {link.channel_identifier,
                 combined_failure_type,
                 link.link_status.traffic_params,
                 link.link_status.metrics.retrain_count,
                 link.link_status.metrics.crc_error_count,
                 link.link_status.metrics.corrected_codeword_count,
                 link.link_status.metrics.uncorrected_codeword_count,
                 link.link_status.num_mismatched_words});
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
        std::cout << "Total Faulty Links: " << link_metrics.size() << std::endl << std::endl;
    }

    // Table header
    std::cout << std::left << std::setw(20) << "Host" << std::setw(6) << "Tray" << std::setw(6) << "ASIC"
              << std::setw(5) << "Ch" << std::setw(9) << "Port ID" << std::setw(15) << "Port Type" << std::setw(14)
              << "Unique ID" << std::setw(12) << "Retrains" << std::setw(14) << "CRC Err" << std::setw(18)
              << "Corrected CW" << std::setw(18) << "Uncorrected CW" << std::setw(16) << "Mismatch Words";

    if (!log_ethernet_metrics) {
        std::cout << std::setw(40) << "Failure Type";
    }

    std::cout << std::setw(12) << "Pkt Size" << std::setw(12) << "Data Size" << std::endl;

    std::cout << std::string(log_ethernet_metrics ? 177 : 217, '-') << std::endl;

    // Table rows
    for (const auto& row : metric_rows) {
        std::cout << std::left << std::setw(20) << row.channel_id.host << std::setw(6) << *row.channel_id.tray_id
                  << std::setw(6) << *row.channel_id.asic_location << std::setw(5)
                  << static_cast<int>(row.channel_id.channel);

        // Print Port ID
        std::cout << std::left << std::setw(9) << row.channel_id.port_id;

        // Print Port Type
        auto port_type = static_cast<tt::scaleout_tools::PortType>(row.channel_id.port_type);
        std::cout << std::left << std::setw(15) << enchantum::to_string(port_type);

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

        // Corrected codewords
        std::stringstream corr_stream;
        corr_stream << "0x" << std::hex << row.corrected_codeword_count;
        std::cout << std::left << std::setw(18) << corr_stream.str();

        // Uncorrected codewords
        std::stringstream uncorr_stream;
        uncorr_stream << "0x" << std::hex << row.uncorrected_codeword_count;
        std::cout << std::left << std::setw(18) << uncorr_stream.str();

        // Mismatched words
        std::cout << std::dec << std::left << std::setw(16) << row.num_mismatched_words;

        // Failure Type (only for faulty links report)
        if (!log_ethernet_metrics) {
            std::cout << std::left << std::setw(40) << row.metric_type;
        }

        std::cout << std::setw(12) << (std::to_string(row.traffic_params.packet_size_bytes) + " B") << std::setw(12)
                  << (std::to_string(row.traffic_params.data_size) + " B") << std::endl;
    }

    std::cout << std::string(log_ethernet_metrics ? 177 : 217, '-') << std::endl << std::endl;

    // Write CSV file
    std::filesystem::path csv_path =
        log_ethernet_metrics ? output_path / "ethernet_metrics_report.csv" : output_path / "unhealthy_links_report.csv";
    std::ofstream csv_file(csv_path);

    if (csv_file.is_open()) {
        // CSV header
        csv_file << "Host,Tray,ASIC,Channel,Port_ID,Port_Type,Unique_ID";
        if (!log_ethernet_metrics) {
            csv_file << ",Failure_Type";
        }
        csv_file << ",Packet_Size_Bytes,Data_Size_Bytes,"
                 << "Retrain_Count,CRC_Error_Count,Corrected_Codeword_Count,Uncorrected_Codeword_Count,Mismatched_Words"
                 << std::endl;

        // CSV rows
        for (const auto& row : metric_rows) {
            auto port_type = static_cast<tt::scaleout_tools::PortType>(row.channel_id.port_type);
            csv_file << row.channel_id.host << "," << *row.channel_id.tray_id << "," << *row.channel_id.asic_location
                     << "," << static_cast<int>(row.channel_id.channel) << "," << row.channel_id.port_id << ","
                     << enchantum::to_string(port_type) << ","
                     << "0x" << std::hex << *row.channel_id.asic_id << std::dec;
            if (!log_ethernet_metrics) {
                csv_file << "," << row.metric_type;
            }
            csv_file << "," << row.traffic_params.packet_size_bytes << "," << row.traffic_params.data_size << ","
                     << row.retrain_count << ","
                     << "0x" << std::hex << row.crc_error_count << std::dec << ","
                     << "0x" << std::hex << row.corrected_codeword_count << std::dec << ","
                     << "0x" << std::hex << row.uncorrected_codeword_count << std::dec << ","
                     << row.num_mismatched_words << std::endl;
        }

        csv_file.close();
        log_output_rank0("✓ Detailed report written to: " + csv_path.string());
    } else {
        log_output_rank0("✗ Warning: Could not open CSV file for writing: " + csv_path.string());
    }
}

void handle_workload_timeout(
    ClusterContext& ctx,
    std::unordered_map<EthChannelIdentifier, std::vector<LinkStatus>>& statuses_per_link,
    std::vector<uint32_t>& inputs,
    size_t data_size,
    size_t packet_size_bytes,
    bool log_ethernet_metrics,
    const ConnectivityValidationConfig& validation_config) {
    log_output_rank0(
        "ERROR: Workload execution timed out after " + std::to_string(WORKLOAD_TIMEOUT_DURATION.count()) +
        " seconds, cluster is not in a healthy state.");

    dump_link_stats(ctx, inputs, statuses_per_link, data_size, packet_size_bytes);
    auto current_result = process_link_statuses(statuses_per_link, true);
    if (log_ethernet_metrics) {
        log_link_metrics(current_result.all_link_metrics, std::filesystem::current_path(), true /*log_all_metrics*/);
    }
    log_link_metrics(current_result.unhealthy_links, std::filesystem::current_path(), false /*log_all_metrics*/);

    if (validation_config.cabling_descriptor_path.has_value() || validation_config.fsd_path.has_value()) {
        log_output_rank0("Re-running discovery to check for link failures");
        ctx.physical_system_descriptor.run_discovery(true, true);

        log_output_rank0("Generating Global System Descriptor in-memory");
        YAML::Node gsd_yaml_node = ctx.physical_system_descriptor.generate_yaml_node();

        log_output_rank0("Obtaining Factory System Descriptor");
        auto fsd_proto = get_factory_system_descriptor(
            validation_config.cabling_descriptor_path,
            validation_config.deployment_descriptor_path,
            validation_config.fsd_path,
            ctx.physical_system_descriptor.get_all_hostnames());
        validate_connectivity(
            fsd_proto, gsd_yaml_node, validation_config.fail_on_warning, ctx.physical_system_descriptor);
    } else {
        log_output_rank0(
            "WARNING: Cannot validate Global System Descriptor against Factory System Descriptor, "
            "no cabling descriptor or factory descriptor provided.");
    }

    // Exit immediately since the cluster is in an unhealthy state
    TT_THROW(
        "Workload execution timed out after {} seconds. Cluster validation failed.", WORKLOAD_TIMEOUT_DURATION.count());
}

LinkMetricsResult send_traffic_and_validate_links(
    PhysicalSystemDescriptor& physical_system_descriptor,
    uint32_t num_iterations,
    bool log_ethernet_metrics,
    bool sweep_traffic_configs,
    uint32_t packet_size_bytes,
    uint32_t data_size,
    std::unordered_map<uint64_t, ChipId>& asic_id_to_chip_id,
    const ConnectivityValidationConfig& validation_config) {
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
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();

    std::vector<ChipId> device_ids;
    for (auto chip : cluster.all_chip_ids()) {
        device_ids.push_back(chip);
    }
    // This is a non-trivial operation, since it loads management firmware onto all
    // cores in the cluster.
    // Issue a global barrier after this to ensure that all hosts in the cluster are ready
    std::map<int, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices = {};
    try {
        devices = tt::tt_metal::distributed::MeshDevice::create_unit_meshes(
            device_ids,
            DEFAULT_L1_SMALL_SIZE,
            DEFAULT_TRACE_REGION_SIZE,
            1,
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config());
    } catch (const std::exception& e) {
        log_info(tt::LogDistributed, "Error starting devices to send traffic on rank: {}", *distributed_context.rank());
        log_output_rank0("Error details: " + std::string(e.what()));
        throw;
    }
    // Barrier here ensures that all ranks successfully started their devices before proceeding
    distributed_context.barrier();

    ClusterContext ctx{physical_system_descriptor, asic_id_to_chip_id, devices};

    std::unordered_map<EthChannelIdentifier, std::vector<LinkStatus>> statuses_per_link;
    bool fwd = true;
    for (int i = 0; i < num_iterations; i++) {
        for (const auto& traffic_config : traffic_configs) {
            std::size_t pkt_size_bytes = traffic_config.packet_size_bytes;
            std::size_t pkt_size_words = pkt_size_bytes >> 4;
            std::size_t d_size = traffic_config.data_size;

            std::unordered_map<ChipId, tt::tt_metal::Program> programs;
            auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, d_size / sizeof(uint32_t));
            configure_local_kernels(ctx, inputs, programs, pkt_size_bytes, pkt_size_words, d_size, fwd);

            configure_cross_host_kernels(ctx, inputs, programs, pkt_size_bytes, pkt_size_words, d_size, fwd);

            WorkloadResult local_result = execute_workloads(programs, devices);
            bool did_hang_locally = (local_result == WorkloadResult::TimedOut);

            // Check if any rank experienced a hang/timeout
            bool any_rank_hung = false;
            distributed_context.all_reduce(
                tt::stl::Span<bool>(&did_hang_locally, 1),
                tt::stl::Span<bool>(&any_rank_hung, 1),
                tt::tt_metal::distributed::multihost::ReduceOp::LOR);

            if (any_rank_hung) {
                handle_workload_timeout(
                    ctx, statuses_per_link, inputs, d_size, pkt_size_bytes, log_ethernet_metrics, validation_config);
            }

            dump_link_stats(ctx, inputs, statuses_per_link, d_size, pkt_size_bytes);
            fwd = !fwd;  // Toggle direction to test bidirectional traffic across links
        }
    }

    return process_link_statuses(statuses_per_link, log_ethernet_metrics);
}

void point_to_point_barrier(const ResetPair& reset_pair) {
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    TT_FATAL(
        *distributed_context.rank() == reset_pair.src_rank || *distributed_context.rank() == reset_pair.dst_rank,
        "Point-to-Point barrier for ranks {} and {} cannot be called on rank {}.",
        reset_pair.src_rank,
        reset_pair.dst_rank,
        *distributed_context.rank());

    uint32_t tag = (reset_pair.src_rank << 8) | reset_pair.dst_rank;

    if (*distributed_context.rank() == reset_pair.src_rank) {
        int sync_msg = 1;
        distributed_context.ssend(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&sync_msg), sizeof(sync_msg)),
            tt::tt_metal::distributed::multihost::Rank{static_cast<int>(reset_pair.dst_rank)},
            tt::tt_metal::distributed::multihost::Tag{static_cast<int>(tag)});
    } else {
        int sync_msg = 0;
        distributed_context.recv(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&sync_msg), sizeof(sync_msg)),
            tt::tt_metal::distributed::multihost::Rank{static_cast<int>(reset_pair.src_rank)},
            tt::tt_metal::distributed::multihost::Tag{static_cast<int>(tag)});
    }
}

void reset_local_link(ChipId src_chip, ChipId dst_chip, uint8_t src_chan, uint8_t dst_chan) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    std::vector<uint32_t> set = {1};

    const auto& sender_soc_desc = cluster.get_soc_desc(src_chip);
    const auto& receiver_soc_desc = cluster.get_soc_desc(dst_chip);
    auto logical_src_coord = sender_soc_desc.get_eth_core_for_channel(src_chan, CoordSystem::LOGICAL);
    auto logical_dst_coord = receiver_soc_desc.get_eth_core_for_channel(dst_chan, CoordSystem::LOGICAL);
    auto src_coord = cluster.get_virtual_coordinate_from_logical_coordinates(
        src_chip, tt_xy_pair(logical_src_coord.x, logical_src_coord.y), CoreType::ETH);
    auto dst_coord = cluster.get_virtual_coordinate_from_logical_coordinates(
        dst_chip, tt_xy_pair(logical_dst_coord.x, logical_dst_coord.y), CoreType::ETH);

    cluster.write_core(src_chip, src_coord, set, 0x1EFC);
    cluster.write_core(dst_chip, dst_coord, set, 0x1EFC);

    bool reset = false;
    while (!reset) {
        std::vector<uint32_t> reset_src = {0};
        std::vector<uint32_t> reset_dst = {0};

        cluster.read_core(reset_src, sizeof(uint32_t), tt_cxy_pair(src_chip, src_coord), 0x1EFC);
        cluster.read_core(reset_dst, sizeof(uint32_t), tt_cxy_pair(dst_chip, dst_coord), 0x1EFC);
        reset = !(reset_src[0] || reset_dst[0]);
    }
}

void forward_link_reset_metadata_from_controller(
    std::unordered_map<uint32_t, std::vector<EthChannelIdentifier>>& ordered_exit_nodes,
    std::unordered_map<uint32_t, std::vector<ResetPair>>& ordered_reset_pairs,
    std::vector<EthChannelIdentifier>& exit_nodes_to_reset,
    std::vector<ResetPair>& reset_pairs) {
    constexpr uint32_t CONTROLLER_RANK = 0;
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();

    if (*distributed_context.rank() == CONTROLLER_RANK) {
        for (const auto& [rank, exit_nodes] : ordered_exit_nodes) {
            if (rank == *distributed_context.rank()) {
                continue;
            }
            auto serialized_exit_nodes = tt::scaleout::validation::serialize_eth_chan_identifiers_to_bytes(exit_nodes);
            auto serialized_exit_nodes_size = serialized_exit_nodes.size();
            auto serialized_reset_pairs =
                tt::scaleout::validation::serialize_reset_pairs_to_bytes(ordered_reset_pairs[rank]);
            auto serialized_reset_pairs_size = serialized_reset_pairs.size();

            distributed_context.send(
                tt::stl::Span<std::byte>(
                    reinterpret_cast<std::byte*>(&serialized_exit_nodes_size), sizeof(serialized_exit_nodes_size)),
                tt::tt_metal::distributed::multihost::Rank{static_cast<int>(rank)},
                tt::tt_metal::distributed::multihost::Tag{0});
            distributed_context.send(
                tt::stl::as_writable_bytes(
                    tt::stl::Span<uint8_t>(serialized_exit_nodes.data(), serialized_exit_nodes.size())),
                tt::tt_metal::distributed::multihost::Rank{static_cast<int>(rank)},
                tt::tt_metal::distributed::multihost::Tag{0});

            distributed_context.send(
                tt::stl::Span<std::byte>(
                    reinterpret_cast<std::byte*>(&serialized_reset_pairs_size), sizeof(serialized_reset_pairs_size)),
                tt::tt_metal::distributed::multihost::Rank{static_cast<int>(rank)},
                tt::tt_metal::distributed::multihost::Tag{0});
            distributed_context.send(
                tt::stl::as_writable_bytes(
                    tt::stl::Span<uint8_t>(serialized_reset_pairs.data(), serialized_reset_pairs.size())),
                tt::tt_metal::distributed::multihost::Rank{static_cast<int>(rank)},
                tt::tt_metal::distributed::multihost::Tag{0});
        }
        exit_nodes_to_reset = ordered_exit_nodes[*distributed_context.rank()];
        reset_pairs = ordered_reset_pairs[*distributed_context.rank()];
    } else {
        std::size_t serialized_exit_nodes_size = 0;
        distributed_context.recv(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(&serialized_exit_nodes_size), sizeof(serialized_exit_nodes_size)),
            tt::tt_metal::distributed::multihost::Rank{CONTROLLER_RANK},
            tt::tt_metal::distributed::multihost::Tag{0});
        std::vector<uint8_t> serialized_exit_nodes(serialized_exit_nodes_size);
        distributed_context.recv(
            tt::stl::as_writable_bytes(
                tt::stl::Span<uint8_t>(serialized_exit_nodes.data(), serialized_exit_nodes.size())),
            tt::tt_metal::distributed::multihost::Rank{CONTROLLER_RANK},
            tt::tt_metal::distributed::multihost::Tag{0});
        exit_nodes_to_reset =
            tt::scaleout::validation::deserialize_eth_chan_identifiers_from_bytes(serialized_exit_nodes);
        std::size_t serialized_reset_pairs_size = 0;
        distributed_context.recv(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(&serialized_reset_pairs_size), sizeof(serialized_reset_pairs_size)),
            tt::tt_metal::distributed::multihost::Rank{CONTROLLER_RANK},
            tt::tt_metal::distributed::multihost::Tag{0});
        std::vector<uint8_t> serialized_reset_pairs(serialized_reset_pairs_size);
        distributed_context.recv(
            tt::stl::as_writable_bytes(
                tt::stl::Span<uint8_t>(serialized_reset_pairs.data(), serialized_reset_pairs.size())),
            tt::tt_metal::distributed::multihost::Rank{CONTROLLER_RANK},
            tt::tt_metal::distributed::multihost::Tag{0});
        reset_pairs = tt::scaleout::validation::deserialize_reset_pairs_from_bytes(serialized_reset_pairs);
    }

    TT_FATAL(
        exit_nodes_to_reset.size() == reset_pairs.size(),
        "Expected reset pairs to be the same size as the number of links to reset {} {} {}",
        exit_nodes_to_reset.size(),
        reset_pairs.size(),
        *distributed_context.rank());
}

void reset_local_ethernet_links(
    const PhysicalSystemDescriptor& physical_system_descriptor, const tt::tt_metal::AsicTopology& asic_topology) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    std::unordered_map<uint64_t, ChipId> asic_id_to_chip_id;

    for (const auto& [chip_id, asic_id] : cluster.get_unique_chip_ids()) {
        asic_id_to_chip_id[asic_id] = chip_id;
    }
    std::vector<uint32_t> set = {1};
    std::unordered_map<uint64_t, std::set<uint8_t>> reset_cores;

    for (const auto& [asic_id, asic_connections] : asic_topology) {
        if (physical_system_descriptor.get_host_name_for_asic(asic_id) != physical_system_descriptor.my_host_name()) {
            continue;
        }
        auto src_chip_id = asic_id_to_chip_id[*asic_id];
        for (const auto& [dst_asic_id, eth_connections] : asic_connections) {
            if (physical_system_descriptor.get_host_name_for_asic(dst_asic_id) !=
                physical_system_descriptor.my_host_name()) {
                continue;
            }
            auto dst_chip_id = asic_id_to_chip_id[*dst_asic_id];
            for (const auto& eth_connection : eth_connections) {
                auto src_chan = eth_connection.src_chan;
                auto dst_chan = eth_connection.dst_chan;

                if (reset_cores[*dst_asic_id].contains(dst_chan)) {
                    TT_FATAL(
                        reset_cores[*asic_id].contains(src_chan),
                        "Expected channel {} on ASIC {} to already be reset",
                        src_chan,
                        *asic_id);
                    continue;
                }
                reset_cores[*asic_id].insert(src_chan);
                reset_cores[*dst_asic_id].insert(dst_chan);
                const auto& asic_descriptor = physical_system_descriptor.get_asic_descriptors().at(asic_id);
                const auto& dst_asic_descriptor = physical_system_descriptor.get_asic_descriptors().at(dst_asic_id);
                log_output_rank0(
                    "Host: " + asic_descriptor.host_name + " Resetting Link " + std::to_string(src_chan) + " on " +
                    " Tray: " + std::to_string(*asic_descriptor.tray_id) + " Location: " +
                    std::to_string(*asic_descriptor.asic_location) + " and Link: " + std::to_string(dst_chan) + " on " +
                    " Tray: " + std::to_string(*dst_asic_descriptor.tray_id) +
                    " Location: " + std::to_string(*dst_asic_descriptor.asic_location));

                reset_local_link(src_chip_id, dst_chip_id, src_chan, dst_chan);
            }
        }
    }
}

void get_cross_node_ethernet_links_to_reset(
    const PhysicalSystemDescriptor& physical_system_descriptor,
    const tt::tt_metal::AsicTopology& asic_topology,
    std::vector<EthChannelIdentifier>& cross_node_links_to_reset,
    std::vector<ResetPair>& cross_node_reset_pairs) {
    constexpr uint32_t CONTROLLER_RANK = 0;

    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();

    std::unordered_map<uint32_t, std::vector<EthChannelIdentifier>> ordered_exit_nodes;
    std::unordered_map<uint32_t, std::vector<ResetPair>> ordered_reset_pairs;
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> paired_asic_ids;

    for (const auto& host : physical_system_descriptor.get_all_hostnames()) {
        ordered_exit_nodes[physical_system_descriptor.get_rank_for_hostname(host)] =
            std::vector<EthChannelIdentifier>();
        ordered_reset_pairs[physical_system_descriptor.get_rank_for_hostname(host)] = std::vector<ResetPair>();
    }

    if (*distributed_context.rank() == CONTROLLER_RANK) {
        for (const auto& [asic_id, asic_connections] : asic_topology) {
            auto src_host_rank = physical_system_descriptor.get_rank_for_hostname(
                physical_system_descriptor.get_host_name_for_asic(asic_id));
            for (const auto& [dst_asic_id, eth_connections] : asic_connections) {
                // These links are being retrained for the second time if the current dst_asic was paired with the
                // current src_asic in a previous iteration.
                // In this case, we skip the link reset.
                if (paired_asic_ids.contains(*dst_asic_id) and paired_asic_ids[*dst_asic_id].contains(*asic_id)) {
                    continue;
                }
                paired_asic_ids[*asic_id].insert(*dst_asic_id);
                auto dst_host_rank = physical_system_descriptor.get_rank_for_hostname(
                    physical_system_descriptor.get_host_name_for_asic(dst_asic_id));
                if (src_host_rank == dst_host_rank) {
                    continue;
                }
                for (const auto& eth_connection : eth_connections) {
                    ordered_exit_nodes[src_host_rank].push_back(EthChannelIdentifier{
                        physical_system_descriptor.get_host_name_for_asic(asic_id),
                        asic_id,
                        tt::tt_metal::TrayID{0},
                        tt::tt_metal::ASICLocation{0},
                        eth_connection.src_chan});
                    ordered_exit_nodes[dst_host_rank].push_back(EthChannelIdentifier{
                        physical_system_descriptor.get_host_name_for_asic(dst_asic_id),
                        dst_asic_id,
                        tt::tt_metal::TrayID{0},
                        tt::tt_metal::ASICLocation{0},
                        eth_connection.dst_chan});
                    ordered_reset_pairs[src_host_rank].push_back(ResetPair{src_host_rank, dst_host_rank});
                    ordered_reset_pairs[dst_host_rank].push_back(ResetPair{src_host_rank, dst_host_rank});
                }
            }
        }
    }
    forward_link_reset_metadata_from_controller(
        ordered_exit_nodes, ordered_reset_pairs, cross_node_links_to_reset, cross_node_reset_pairs);
}

void reset_cross_node_ethernet_links(
    const PhysicalSystemDescriptor& physical_system_descriptor,
    const std::vector<EthChannelIdentifier>& cross_node_links_to_reset,
    const std::vector<ResetPair>& cross_node_reset_pairs) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    std::vector<uint32_t> set = {1};

    std::unordered_map<uint64_t, ChipId> asic_id_to_chip_id;

    for (const auto& [chip_id, asic_id] : cluster.get_unique_chip_ids()) {
        asic_id_to_chip_id[asic_id] = chip_id;
    }

    for (size_t i = 0; i < cross_node_links_to_reset.size(); i++) {
        const auto& link = cross_node_links_to_reset[i];
        const auto& reset_pair = cross_node_reset_pairs[i];

        auto src_chip_id = asic_id_to_chip_id[*link.asic_id];
        const auto& src_soc_desc = cluster.get_soc_desc(src_chip_id);
        auto logical_src_coord = src_soc_desc.get_eth_core_for_channel(link.channel, CoordSystem::LOGICAL);
        auto src_coord = cluster.get_virtual_coordinate_from_logical_coordinates(
            src_chip_id, tt_xy_pair(logical_src_coord.x, logical_src_coord.y), CoreType::ETH);
        const auto& asic_descriptor = physical_system_descriptor.get_asic_descriptors().at(link.asic_id);
        log_output_rank0(
            "Resetting Cross-Node Link " + std::to_string(link.channel) + " on " + std::to_string(*link.asic_id) +
            " Host: " + asic_descriptor.host_name + " Tray: " + std::to_string(*asic_descriptor.tray_id) +
            " Location: " + std::to_string(*asic_descriptor.asic_location));
        point_to_point_barrier(reset_pair);
        cluster.write_core(src_chip_id, src_coord, set, 0x1EFC);
        std::vector<uint32_t> reset = {1};
        while (reset[0]) {
            cluster.read_core(reset, sizeof(uint32_t), tt_cxy_pair(src_chip_id, src_coord), 0x1EFC);
        }
    }
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    // Barrier ensures all hosts have completed their cross-node ethernet link resets before proceeding.
    // This is critical because cross-node resets involve coordination between paired hosts.
    distributed_context.barrier();
}

void reset_ethernet_links(
    const PhysicalSystemDescriptor& physical_system_descriptor, const tt::tt_metal::AsicTopology& asic_topology) {
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    // Reset All Local Ethernet Links, specified in the topology. Ethernet Links on Exit Nodes are reset separately.
    reset_local_ethernet_links(physical_system_descriptor, asic_topology);
    // Barrier ensures all hosts have completed local link resets before starting cross-node resets.
    // This prevents race conditions where one host might start cross-node reset while another is still
    // resetting local links.
    distributed_context.barrier();

    // Reset All Cross-Node Ethernet Links, specified in the topology.
    std::vector<EthChannelIdentifier> cross_node_links_to_reset;
    std::vector<ResetPair> cross_node_reset_pairs;
    get_cross_node_ethernet_links_to_reset(
        physical_system_descriptor, asic_topology, cross_node_links_to_reset, cross_node_reset_pairs);
    reset_cross_node_ethernet_links(physical_system_descriptor, cross_node_links_to_reset, cross_node_reset_pairs);
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
    const ConnectivityValidationConfig& validation_config) {
    std::unordered_map<uint64_t, ChipId> asic_id_to_chip_id;
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
            asic_id_to_chip_id,
            validation_config);
    } else if (log_ethernet_metrics) {
        std::unordered_map<EthChannelIdentifier, std::vector<LinkStatus>> statuses_per_link;
        std::vector<uint32_t> inputs = {};
        std::map<int, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> empty_devices;
        ClusterContext ctx{physical_system_descriptor, asic_id_to_chip_id, empty_devices};
        dump_link_stats(ctx, inputs, statuses_per_link, 0 /*data_size*/, 0 /*packet_size_bytes*/);
        result = process_link_statuses(statuses_per_link, log_ethernet_metrics);
    }

    // Log metrics
    if (log_ethernet_metrics) {
        // Log all ethernet metrics
        log_link_metrics(result.all_link_metrics, validation_config.output_path, true /*log_all_metrics*/);
    }
    log_link_metrics(result.unhealthy_links, validation_config.output_path, false /*log_all_metrics*/);
    return result.unhealthy_links.empty();
}

tt::tt_metal::AsicTopology generate_asic_topology_from_connections(
    const std::set<PhysicalChannelConnection>& physical_connections,
    PhysicalSystemDescriptor& physical_system_descriptor) {
    tt::tt_metal::AsicTopology asic_topology;
    std::unordered_map<tt_metal::AsicID, std::set<tt_metal::AsicID>> visited;
    std::unordered_map<tt_metal::AsicID, std::unordered_map<tt_metal::AsicID, uint32_t>> visited_idx;
    for (const auto& connection : physical_connections) {
        auto src = connection.first;
        auto dst = connection.second;
        auto src_asic_id = physical_system_descriptor.get_asic_id(
            src.hostname, tt::tt_metal::TrayID(*src.tray_id), tt_metal::ASICLocation(src.asic_channel.asic_location));
        auto dst_asic_id = physical_system_descriptor.get_asic_id(
            dst.hostname, tt::tt_metal::TrayID(*dst.tray_id), tt_metal::ASICLocation(dst.asic_channel.asic_location));
        if (!visited[src_asic_id].contains(dst_asic_id)) {
            asic_topology[src_asic_id].push_back(
                {dst_asic_id,
                 {tt::tt_metal::EthConnection(
                     *src.asic_channel.channel_id, *dst.asic_channel.channel_id, src.hostname == dst.hostname)}});
            visited[src_asic_id].insert(dst_asic_id);
            visited_idx[src_asic_id][dst_asic_id] = asic_topology[src_asic_id].size() - 1;
        } else {
            asic_topology[src_asic_id][visited_idx[src_asic_id][dst_asic_id]].second.push_back(
                tt::tt_metal::EthConnection(
                    *src.asic_channel.channel_id, *dst.asic_channel.channel_id, src.hostname == dst.hostname));
        }
        if (!visited[dst_asic_id].contains(src_asic_id)) {
            asic_topology[dst_asic_id].push_back(
                {src_asic_id,
                 {tt::tt_metal::EthConnection(
                     *dst.asic_channel.channel_id, *src.asic_channel.channel_id, src.hostname == dst.hostname)}});
            visited[dst_asic_id].insert(src_asic_id);
            visited_idx[dst_asic_id][src_asic_id] = asic_topology[dst_asic_id].size() - 1;
        } else {
            asic_topology[dst_asic_id][visited_idx[dst_asic_id][src_asic_id]].second.push_back(
                tt::tt_metal::EthConnection(
                    *dst.asic_channel.channel_id, *src.asic_channel.channel_id, src.hostname == dst.hostname));
        }
    }
    return asic_topology;
}

tt::tt_metal::AsicTopology build_reset_topology(
    const std::string& reset_host,
    uint32_t reset_tray_id,
    uint32_t reset_asic_location,
    uint32_t reset_channel,
    PhysicalSystemDescriptor& physical_system_descriptor) {
    log_output_rank0("Building reset topology for specified link");
    log_output_rank0("  Host: " + reset_host);
    log_output_rank0("  Tray ID: " + std::to_string(reset_tray_id));
    log_output_rank0("  ASIC Location: " + std::to_string(reset_asic_location));
    log_output_rank0("  Channel: " + std::to_string(reset_channel));

    tt::tt_metal::AsicID src_asic_id = physical_system_descriptor.get_asic_id(
        reset_host, tt::tt_metal::TrayID(reset_tray_id), tt::tt_metal::ASICLocation(reset_asic_location));
    uint8_t src_channel = static_cast<uint8_t>(reset_channel);

    auto [dst_asic_id, dst_channel] =
        physical_system_descriptor.get_connected_asic_and_channel(src_asic_id, src_channel);

    const auto& asic_descriptors = physical_system_descriptor.get_asic_descriptors();
    TT_FATAL(
        asic_descriptors.contains(dst_asic_id),
        "Could not find ASIC descriptor for destination ASIC ID: {}",
        dst_asic_id);

    const auto& dst_asic_descriptor = asic_descriptors.at(dst_asic_id);
    std::string dst_host = dst_asic_descriptor.host_name;
    bool is_local = (reset_host == dst_host);

    log_output_rank0("  Discovered Destination:");
    log_output_rank0("    Host: " + dst_host);
    log_output_rank0("    Tray ID: " + std::to_string(*dst_asic_descriptor.tray_id));
    log_output_rank0("    ASIC Location: " + std::to_string(*dst_asic_descriptor.asic_location));
    log_output_rank0("    Channel: " + std::to_string(dst_channel));
    log_output_rank0("  Connection Type: " + std::string(is_local ? "Local" : "Remote"));

    tt::tt_metal::AsicTopology asic_topology;

    tt::tt_metal::EthConnection src_to_dst_conn;
    src_to_dst_conn.src_chan = src_channel;
    src_to_dst_conn.dst_chan = dst_channel;
    src_to_dst_conn.is_local = is_local;

    tt::tt_metal::EthConnection dst_to_src_conn;
    dst_to_src_conn.src_chan = dst_channel;
    dst_to_src_conn.dst_chan = src_channel;
    dst_to_src_conn.is_local = is_local;

    asic_topology[src_asic_id].push_back({dst_asic_id, {src_to_dst_conn}});
    asic_topology[dst_asic_id].push_back({src_asic_id, {dst_to_src_conn}});

    log_output_rank0("Reset topology built successfully");

    return asic_topology;
}

void perform_link_reset(
    const std::string& reset_host,
    uint32_t reset_tray_id,
    uint32_t reset_asic_location,
    uint32_t reset_channel,
    PhysicalSystemDescriptor& physical_system_descriptor) {
    bool link_retrain_supported = tt::tt_metal::MetalContext::instance().get_cluster().arch() == tt::ARCH::WORMHOLE_B0;
    TT_FATAL(link_retrain_supported, "Link reset is only supported on WORMHOLE_B0 architecture");

    tt::tt_metal::AsicTopology reset_topology =
        build_reset_topology(reset_host, reset_tray_id, reset_asic_location, reset_channel, physical_system_descriptor);

    reset_ethernet_links(physical_system_descriptor, reset_topology);

    log_output_rank0("Link reset completed. Please run the validation tool again to verify the link.");
}

fsd::proto::FactorySystemDescriptor get_factory_system_descriptor(
    const std::optional<std::string>& cabling_descriptor_path,
    const std::optional<std::string>& deployment_descriptor_path,
    const std::optional<std::string>& fsd_path,
    const std::vector<std::string>& hostnames) {
    if (!cabling_descriptor_path.has_value() && !fsd_path.has_value()) {
        TT_THROW("Either cabling_descriptor_path or fsd_path must be provided");
    }

    if (cabling_descriptor_path.has_value()) {
        if (fsd_path.has_value()) {
            log_warning(
                tt::LogDistributed,
                "Both cabling_descriptor_path and fsd_path provided; using cabling_descriptor_path to generate FSD");
        }
        log_output_rank0("Creating Factory System Descriptor (Golden Representation)");
        if (!deployment_descriptor_path.has_value()) {
            TT_FATAL(
                hostnames.size() == 1,
                "Expected exactly one host in the cluster when no deployment descriptor is provided");
            return tt::scaleout_tools::CablingGenerator(cabling_descriptor_path.value(), hostnames)
                .generate_factory_system_descriptor();
        }
        return tt::scaleout_tools::CablingGenerator(cabling_descriptor_path.value(), deployment_descriptor_path.value())
            .generate_factory_system_descriptor();

    }  // Load FSD from file
    fsd::proto::FactorySystemDescriptor fsd_proto;
    std::ifstream fsd_file(fsd_path.value());
    if (!fsd_file.is_open()) {
        TT_THROW("Failed to open FSD file: {}", fsd_path.value());
    }
    std::string fsd_content((std::istreambuf_iterator<char>(fsd_file)), std::istreambuf_iterator<char>());
    fsd_file.close();
    if (!google::protobuf::TextFormat::ParseFromString(fsd_content, &fsd_proto)) {
        TT_THROW("Failed to parse FSD protobuf from file: {}", fsd_path.value());
    }
    return fsd_proto;
}

tt::tt_metal::AsicTopology validate_connectivity(
    const fsd::proto::FactorySystemDescriptor& fsd_proto,
    const YAML::Node& gsd_yaml_node,
    bool fail_on_warning,
    PhysicalSystemDescriptor& physical_system_descriptor,
    std::optional<uint32_t> min_connections) {
    log_output_rank0(
        "Validating Factory System Descriptor (Golden Representation) against Global System Descriptor (in-memory)");
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    auto missing_physical_connections = tt::scaleout_tools::validate_fsd_against_gsd(
        fsd_proto,
        gsd_yaml_node,
        true /* strict_validation */,
        fail_on_warning,
        *distributed_context.rank() == 0 /* log_output */,
        min_connections);
    log_output_rank0("Factory System Descriptor (Golden Representation) Validation Complete");
    return generate_asic_topology_from_connections(missing_physical_connections, physical_system_descriptor);
}

}  // namespace tt::scaleout_tools
