// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Make traffic bidirectional
// Monitor increasing retrain count: CRC should be zero (or should not be incrementing). Non zero uncorrected words
// Sanity check current 150K is okay - stress 1G or more

// Reset + validaiton Sequence (WH):
//  - Check if specific link came up
// If not, directed reset
// Keep looping until all links are up
// Send traffic
// Check stats
// If link unhealthy, issue a directed reset again
// Give up at some point

#include <iostream>
#include <string>
#include <iomanip>
#include <fstream>

#include <factory_system_descriptor/utils.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt-metalium/distributed.hpp>
#include "tt_metal/impl/context/metal_context.hpp"
#include "tests/tt_metal/test_utils/test_common.hpp"
#include <cabling_generator/cabling_generator.hpp>
#include <tt-metalium/hal.hpp>
#include <algorithm>
#include <random>
#include "tools/scaleout/validation/faulty_links.hpp"
#include "tools/scaleout/validation/faulty_links_serialization.hpp"

template <typename T1, typename T2>
constexpr std::common_type_t<T1, T2> align_down(T1 value, T2 alignment) {
    static_assert(std::is_integral<T1>::value, "align_down() requires integral types");
    static_assert(std::is_integral<T2>::value, "align_down() requires integral types");
    using T = std::common_type_t<T1, T2>;
    return static_cast<T>(value) & ~(static_cast<T>(alignment) - 1);
}

void log_output_rank0(const std::string& message) {
    if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 0) {
        log_info(tt::LogDistributed, "{}", message);
    }
}

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
    bool send_traffic = false;
    uint32_t data_size = align_down(tt::tt_metal::hal::get_erisc_l1_unreserved_size(), 64);
    uint32_t packet_size_bytes = 64;
    uint32_t num_iterations = 50;
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

struct TestResults {
    tt::tt_metal::EthernetMetrics metrics;
    TestParams test_params;
    uint32_t num_mismatched_words = 0;
};

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
                        {src_eth_l1_byte_address, dst_eth_l1_byte_address, data_size, 0});

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
                    my_program,
                    sender_kernel,
                    my_coord,
                    {src_eth_l1_byte_address, dst_eth_l1_byte_address, data_size, 0});
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
    std::map<int, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices) {
    std::unordered_map<chip_id_t, tt::tt_metal::distributed::MeshWorkload> mesh_workloads;

    for (auto& [device_id, program] : programs) {
        mesh_workloads[device_id] = tt::tt_metal::distributed::MeshWorkload();
        mesh_workloads[device_id].add_program(
            tt::tt_metal::distributed::MeshCoordinateRange(
                tt::tt_metal::distributed::MeshCoordinate(0, 0), tt::tt_metal::distributed::MeshCoordinate(0, 0)),
            std::move(program));
    }
    std::thread threads[mesh_workloads.size()];
    for (auto& [device_id, mesh_workload] : mesh_workloads) {
        threads[device_id] = std::thread([&]() {
            tt::tt_metal::distributed::EnqueueMeshWorkload(
                devices[device_id]->mesh_command_queue(), mesh_workload, true);
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

bool link_unhealthy(const std::vector<TestResults>& link_stats, LinkFailure& failure_mode) {
    auto retrain_count_increasing = [&](const std::vector<TestResults>& link_stats) {
        uint32_t prev_retrain_count = 0;
        for (const auto& dumped_stat : link_stats) {
            const auto& metric = dumped_stat.metrics;
            if (metric.retrain_count > prev_retrain_count) {
                failure_mode.retrain_failure.retrain_count = metric.retrain_count;
                failure_mode.retrain_failure.test_params = dumped_stat.test_params;
                return true;
            }
            prev_retrain_count = metric.retrain_count;
        }
        return false;
    };

    auto crc_error_reported = [&](const std::vector<TestResults>& link_stats) {
        return std::any_of(link_stats.begin(), link_stats.end(), [&](const TestResults& dumped_stat) {
            if (dumped_stat.metrics.crc_error_count > 0) {
                failure_mode.crc_error_failure.crc_error_count = dumped_stat.metrics.crc_error_count;
                failure_mode.crc_error_failure.test_params = dumped_stat.test_params;
                return true;
            }
            return false;
        });
    };

    auto uncorrected_codewords_detected = [&](const std::vector<TestResults>& link_stats) {
        return std::any_of(link_stats.begin(), link_stats.end(), [&](const TestResults& dumped_stat) {
            if (dumped_stat.metrics.uncorrected_codeword_count > 0) {
                failure_mode.uncorrected_codeword_failure.uncorrected_codeword_count =
                    dumped_stat.metrics.uncorrected_codeword_count;
                failure_mode.uncorrected_codeword_failure.test_params = dumped_stat.test_params;
                return true;
            }
            return false;
        });
    };

    auto data_mismatch = [&](const std::vector<TestResults>& link_stats) {
        return std::any_of(link_stats.begin(), link_stats.end(), [&](const TestResults& dumped_stat) {
            if (dumped_stat.num_mismatched_words > 0) {
                failure_mode.data_mismatch_failure.num_mismatched_words = dumped_stat.num_mismatched_words;
                failure_mode.data_mismatch_failure.test_params = dumped_stat.test_params;
                return true;
            }
            return false;
        });
    };

    return retrain_count_increasing(link_stats) || crc_error_reported(link_stats) ||
           uncorrected_codewords_detected(link_stats) || data_mismatch(link_stats);
}

void dump_link_stats(
    std::vector<uint32_t>& inputs,
    PhysicalSystemDescriptor& physical_system_descriptor,
    std::map<int, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices,
    std::unordered_map<uint64_t, chip_id_t>& asic_id_to_chip_id,
    size_t data_size,
    size_t packet_size_bytes,
    std::unordered_map<EthChannelIdentifier, std::vector<TestResults>>& test_results) {
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
                auto result_vec = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                    chip_id,
                    devices[chip_id]->ethernet_core_from_logical_core(coord),
                    src_eth_l1_byte_address,
                    data_size);

                // Count mismatched words
                uint32_t num_mismatched = 0;
                for (size_t i = 0; i < result_vec.size(); ++i) {
                    if (result_vec[i] != inputs[i]) {
                        num_mismatched++;
                    }
                }

                test_results[EthChannelIdentifier{
                                 .host = host_name,
                                 .asic_id = asic_descriptors.at(asic_id).unique_id,
                                 .tray_id = asic_descriptors.at(asic_id).tray_id,
                                 .asic_location = asic_descriptors.at(asic_id).asic_location,
                                 .channel = src_chan,
                             }]
                    .push_back(TestResults{
                        .metrics = physical_system_descriptor.get_ethernet_metrics().at(asic_id).at(src_chan),
                        .test_params =
                            TestParams{
                                .packet_size_bytes = packet_size_bytes,
                                .data_size = data_size,
                            },
                        .num_mismatched_words = num_mismatched,
                    });
            }
        }
    }
}

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

void forward_faulty_link_list_to_controller(std::vector<FaultyLink>& faulty_links) {
    constexpr uint32_t CONTROLLER_RANK = 0;
    auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    auto my_rank = *distributed_context.rank();

    std::vector<uint8_t> serialized_faulty_link_list;
    std::size_t serialized_faulty_link_list_size = 0;

    if (my_rank != CONTROLLER_RANK) {
        serialized_faulty_link_list = tt::scaleout::validation::serialize_faulty_links_to_bytes(faulty_links);
        serialized_faulty_link_list_size = serialized_faulty_link_list.size();
        distributed_context.send(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(&serialized_faulty_link_list_size),
                sizeof(serialized_faulty_link_list_size)),
            tt::tt_metal::distributed::multihost::Rank{CONTROLLER_RANK},
            tt::tt_metal::distributed::multihost::Tag{0});
        distributed_context.send(
            tt::stl::as_writable_bytes(
                tt::stl::Span<uint8_t>(serialized_faulty_link_list.data(), serialized_faulty_link_list.size())),
            tt::tt_metal::distributed::multihost::Rank{CONTROLLER_RANK},
            tt::tt_metal::distributed::multihost::Tag{0});
    } else {
        for (auto peer_rank = 0; peer_rank < *distributed_context.size(); peer_rank++) {
            if (peer_rank == CONTROLLER_RANK) {
                continue;
            }
            distributed_context.recv(
                tt::stl::Span<std::byte>(
                    reinterpret_cast<std::byte*>(&serialized_faulty_link_list_size),
                    sizeof(serialized_faulty_link_list_size)),
                tt::tt_metal::distributed::multihost::Rank{peer_rank},
                tt::tt_metal::distributed::multihost::Tag{0});
            serialized_faulty_link_list.resize(serialized_faulty_link_list_size);
            std::cout << "Receiving faulty link list from controller" << std::endl;
            distributed_context.recv(
                tt::stl::as_writable_bytes(
                    tt::stl::Span<uint8_t>(serialized_faulty_link_list.data(), serialized_faulty_link_list.size())),
                tt::tt_metal::distributed::multihost::Rank{peer_rank},
                tt::tt_metal::distributed::multihost::Tag{0});
            std::cout << "Deserializing faulty link list from controller" << std::endl;
            std::vector<FaultyLink> remote_faulty_links =
                tt::scaleout::validation::deserialize_faulty_links_from_bytes(serialized_faulty_link_list);
            faulty_links.insert(faulty_links.end(), remote_faulty_links.begin(), remote_faulty_links.end());
        }
    }
}

std::vector<FaultyLink> send_traffic_and_validate_links(
    PhysicalSystemDescriptor& physical_system_descriptor, const InputArgs& input_args) {
    uint32_t packet_size_bytes = input_args.packet_size_bytes;
    uint32_t packet_size_words = input_args.packet_size_bytes >> 4;
    uint32_t data_size = input_args.data_size;
    uint32_t num_iterations = input_args.num_iterations;

    log_output_rank0(
        "Sending traffic across detected links. Num Iterations: " + std::to_string(num_iterations) +
        " Packet Size (Bytes): " + std::to_string(packet_size_bytes) +
        " Total Data Size: (Bytes): " + std::to_string(data_size));

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

    std::unordered_map<uint64_t, chip_id_t> asic_id_to_chip_id;
    for (const auto& [chip_id, asic_id] : cluster.get_unique_chip_ids()) {
        asic_id_to_chip_id[asic_id] = chip_id;
    }

    std::unordered_map<EthChannelIdentifier, std::vector<TestResults>> test_results;
    std::vector<FaultyLink> faulty_links;
    for (int i = 0; i < num_iterations; i++) {
        std::unordered_map<chip_id_t, tt::tt_metal::Program> programs;
        auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, data_size / sizeof(uint32_t));

        configure_local_kernels(
            physical_system_descriptor,
            asic_id_to_chip_id,
            devices,
            inputs,
            programs,
            packet_size_bytes,
            packet_size_words,
            data_size);

        configure_cross_host_kernels(
            physical_system_descriptor,
            asic_id_to_chip_id,
            devices,
            inputs,
            programs,
            packet_size_bytes,
            packet_size_words,
            data_size);

        execute_workloads(programs, devices);

        dump_link_stats(
            inputs,
            physical_system_descriptor,
            devices,
            asic_id_to_chip_id,
            data_size,
            packet_size_bytes,
            test_results);
    }

    for (const auto& [channel_identifier, link_stats] : test_results) {
        LinkFailure failure_mode;
        if (link_unhealthy(link_stats, failure_mode)) {
            faulty_links.push_back(FaultyLink{
                .channel_identifier = channel_identifier,
                .link_failure = failure_mode,
            });
        }
    }
    std::cout << "Forwarding faulty link list to controller" << std::endl;
    forward_faulty_link_list_to_controller(faulty_links);
    std::cout << "Faulty link list forwarded to controller" << std::endl;
    return faulty_links;
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
    input_args.send_traffic = test_args::has_command_option(args_vec, "--send-traffic");
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
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        const auto& distributed_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
        const auto& hal_ptr = tt::tt_metal::MetalContext::instance().hal_ptr();
        constexpr bool run_discovery = true;
        auto physical_system_descriptor = tt::tt_metal::PhysicalSystemDescriptor(
            cluster.get_driver(), distributed_context, hal_ptr, false, run_discovery);
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

std::string get_port_type_str(const tt::scaleout_tools::PortType& port_type) {
    switch (port_type) {
        case tt::scaleout_tools::PortType::TRACE: return "TRACE";
        case tt::scaleout_tools::PortType::QSFP_DD: return "QSFP_DD";
        case tt::scaleout_tools::PortType::WARP100: return "WARP100";
        case tt::scaleout_tools::PortType::WARP400: return "WARP400";
        case tt::scaleout_tools::PortType::LINKING_BOARD_1: return "LINKING_BOARD_1";
        case tt::scaleout_tools::PortType::LINKING_BOARD_2: return "LINKING_BOARD_2";
        case tt::scaleout_tools::PortType::LINKING_BOARD_3: return "LINKING_BOARD_3";
        default: return "UNKNOWN";
    }
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
    std::map<bool, std::map<std::string, std::map<std::string, std::vector<ConnectionInfo>>>> organized_connections;

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
                    auto port_type_str = get_port_type_str(port_types.at(asic_id).at(channel));

                    ConnectionInfo conn_info{
                        asic_id,
                        channel,
                        host,
                        tray_id,
                        asic_location,
                        port_types.at(asic_id).at(channel),
                        connected_asic_id,
                        connected_channel,
                        connected_host,
                        connected_tray_id,
                        connected_asic_location,
                        metrics};

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

void log_faulty_links(const std::vector<FaultyLink>& faulty_links, const std::filesystem::path& output_path) {
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();

    if (*distributed_context.rank() != 0) {
        return;
    }

    if (faulty_links.empty()) {
        log_output_rank0("All Detected Links are healthy");
        return;
    }

    log_output_rank0(
        "Generating Faulty Link Report (Only Pinging Detected Links). Num Faulty Links: " +
        std::to_string(faulty_links.size()));

    struct FailureRow {
        EthChannelIdentifier channel_id;
        std::string failure_type;
        TestParams test_params;
        uint32_t retrain_count;
        uint32_t crc_error_count;
        uint32_t uncorrected_codeword_count;
        uint32_t num_mismatched_words;
    };

    // Expand each link into separate rows per failure type
    std::vector<FailureRow> failure_rows;
    for (const auto& link : faulty_links) {
        if (link.link_failure.retrain_failure.retrain_count > 0) {
            failure_rows.push_back(
                {link.channel_identifier,
                 "Retrain",
                 link.link_failure.retrain_failure.test_params,
                 link.link_failure.retrain_failure.retrain_count,
                 0,
                 0,
                 0});
        }
        if (link.link_failure.crc_error_failure.crc_error_count > 0) {
            failure_rows.push_back(
                {link.channel_identifier,
                 "CRC Error",
                 link.link_failure.crc_error_failure.test_params,
                 0,
                 link.link_failure.crc_error_failure.crc_error_count,
                 0,
                 0});
        }
        if (link.link_failure.uncorrected_codeword_failure.uncorrected_codeword_count > 0) {
            failure_rows.push_back(
                {link.channel_identifier,
                 "Uncorrected CW",
                 link.link_failure.uncorrected_codeword_failure.test_params,
                 0,
                 0,
                 link.link_failure.uncorrected_codeword_failure.uncorrected_codeword_count,
                 0});
        }
        if (link.link_failure.data_mismatch_failure.num_mismatched_words > 0) {
            failure_rows.push_back(
                {link.channel_identifier,
                 "Data Mismatch",
                 link.link_failure.data_mismatch_failure.test_params,
                 0,
                 0,
                 0,
                 link.link_failure.data_mismatch_failure.num_mismatched_words});
        }
    }

    // Print console table
    std::cout << std::endl;
    std::cout << "╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗"
              << std::endl;
    std::cout << "║                              FAULTY LINKS REPORT                                                  ║"
              << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝"
              << std::endl;
    std::cout << "Total Faulty Link Occurrences: " << faulty_links.size() << std::endl;
    std::cout << "Total Failure Instances: " << failure_rows.size() << std::endl << std::endl;

    // Table header
    std::cout << std::left << std::setw(20) << "Host" << std::setw(6) << "Tray" << std::setw(6) << "ASIC"
              << std::setw(5) << "Ch" << std::setw(14) << "Unique ID" << std::setw(12) << "Retrains" << std::setw(14)
              << "CRC Err" << std::setw(18) << "Uncorrected CW" << std::setw(16) << "Mismatch Words" << std::setw(18)
              << "Failure Type" << std::setw(12) << "Pkt Size" << std::setw(12) << "Data Size" << std::endl;

    std::cout << std::string(153, '-') << std::endl;

    // Table rows
    for (const auto& row : failure_rows) {
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

        std::cout << std::left << std::setw(18) << row.failure_type << std::setw(12)
                  << (std::to_string(row.test_params.packet_size_bytes) + " B") << std::setw(12)
                  << (std::to_string(row.test_params.data_size) + " B") << std::endl;
    }

    std::cout << std::string(153, '-') << std::endl << std::endl;

    // Write CSV file
    std::filesystem::path csv_path = output_path / "faulty_links_report.csv";
    std::ofstream csv_file(csv_path);

    if (csv_file.is_open()) {
        // CSV header
        csv_file << "Host,Tray,ASIC,Channel,Unique_ID,Failure_Type,"
                 << "Packet_Size_Bytes,Data_Size_Bytes,"
                 << "Retrain_Count,CRC_Error_Count,Uncorrected_Codeword_Count,Mismatched_Words" << std::endl;

        // CSV rows
        for (const auto& row : failure_rows) {
            csv_file << row.channel_id.host << "," << *row.channel_id.tray_id << "," << *row.channel_id.asic_location
                     << "," << static_cast<int>(row.channel_id.channel) << ","
                     << "0x" << std::hex << *row.channel_id.asic_id << std::dec << "," << row.failure_type << ","
                     << row.test_params.packet_size_bytes << "," << row.test_params.data_size << ","
                     << row.retrain_count << ","
                     << "0x" << std::hex << row.crc_error_count << std::dec << ","
                     << "0x" << std::hex << row.uncorrected_codeword_count << std::dec << ","
                     << row.num_mismatched_words << std::endl;
        }

        csv_file.close();
        std::cout << "✓ Detailed report written to: " << csv_path << std::endl << std::endl;
    } else {
        std::cerr << "✗ Warning: Could not open CSV file for writing: " << csv_path << std::endl;
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

    if (*distributed_context.rank() == 0) {
        // Set output path for the YAML file
        std::string gsd_yaml_path = input_args.output_path / "global_system_descriptor.yaml";
        // Dump the discovered system to YAML
        physical_system_descriptor.dump_to_yaml(gsd_yaml_path);
        log_output_rank0(
            "Validating Factory System Descriptor (Golden Representation) against Global System Descriptor");
        tt::scaleout_tools::validate_fsd_against_gsd(
            get_factory_system_descriptor_path(input_args), gsd_yaml_path, true, input_args.fail_on_warning);
        log_output_rank0("Factory System Descriptor (Golden Representation) Validation Complete");
    }

    if (input_args.send_traffic) {
        auto faulty_links = send_traffic_and_validate_links(physical_system_descriptor, input_args);
        log_faulty_links(faulty_links, input_args.output_path);
    }

    if (*distributed_context.rank() == 0) {
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
