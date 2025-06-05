// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <gtest/gtest.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <string_view>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/allocator.hpp>
#include "fabric_fixture.hpp"
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include "tt_metal/fabric/fabric_mux_config.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric {
namespace fabric_router_tests {
namespace fabric_mux_tests {

const std::string mux_kernel_src = "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp";
const std::string sender_kernel_src =
    "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_mux_sender_client.cpp";
const std::string receiver_kernel_src =
    "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_mux_receiver_client.cpp";

const uint32_t test_results_size_bytes = 128;
const uint32_t noc_address_padding_bytes = 16;
const uint32_t packet_header_buffer_size_bytes = 1024;

const auto routing_directions = {
    tt_fabric::RoutingDirection::N,
    tt_fabric::RoutingDirection::S,
    tt_fabric::RoutingDirection::E,
    tt_fabric::RoutingDirection::W};

using FabricMuxFixture = Fabric1DFixture;

struct TestConfig {
    uint32_t num_devices = 0;
    uint32_t num_sender_clients = 0;
    uint32_t num_packets = 0;
    uint32_t num_credits = 0;
    uint32_t num_return_credits_per_packet = 0;
    uint32_t packet_payload_size_bytes = 0;
    uint32_t time_seed = 0;
    uint8_t num_buffers_full_size_channel = 0;
    uint8_t num_buffers_header_only_channel = 0;
    bool uniform_sender_receiver_split = true;
    uint32_t num_open_close_iters = 0;
    uint32_t num_full_size_channel_iters = 0;
};

struct WorkerMemoryMap {
    uint32_t test_results_address = 0;
    uint32_t local_mux_status_address = 0;
    uint32_t local_flow_control_address = 0;
    uint32_t local_teardown_address = 0;
    uint32_t local_buffer_index_address = 0;
    uint32_t credit_handshake_address = 0;
    uint32_t packet_header_buffer_address = 0;
    uint32_t base_l1_target_address = 0;
    uint32_t payload_buffer_address = 0;
};

struct WorkerTestConfig {
    WorkerMemoryMap* memory_map = nullptr;
    CoreCoord worker_logical_core;
    uint8_t worker_id = 0;
    tt::tt_fabric::FabricMuxChannelType channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
    uint32_t num_buffers = 0;
    uint32_t buffer_size_bytes = 0;
    uint8_t num_hops = 0;
    std::string_view kernel_src = sender_kernel_src;
};

WorkerMemoryMap create_worker_memory_map(const uint32_t base_l1_address) {
    WorkerMemoryMap worker_memory_map;

    worker_memory_map.test_results_address = base_l1_address;
    worker_memory_map.local_mux_status_address = worker_memory_map.test_results_address + test_results_size_bytes;
    worker_memory_map.local_flow_control_address =
        worker_memory_map.local_mux_status_address + noc_address_padding_bytes;
    worker_memory_map.local_teardown_address = worker_memory_map.local_flow_control_address + noc_address_padding_bytes;
    worker_memory_map.local_buffer_index_address = worker_memory_map.local_teardown_address + noc_address_padding_bytes;
    worker_memory_map.credit_handshake_address =
        worker_memory_map.local_buffer_index_address + noc_address_padding_bytes;
    worker_memory_map.packet_header_buffer_address =
        worker_memory_map.credit_handshake_address + noc_address_padding_bytes;
    worker_memory_map.base_l1_target_address =
        worker_memory_map.packet_header_buffer_address + packet_header_buffer_size_bytes;
    worker_memory_map.payload_buffer_address = worker_memory_map.base_l1_target_address;

    return worker_memory_map;
}

// first generates the physical chip id matrix and then returns the sequence of connected chip ids
std::vector<chip_id_t> get_physical_chip_sequence(uint32_t num_seq_chips) {
    auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    tt::tt_fabric::MeshId mesh_id = control_plane.get_user_physical_mesh_ids()[0];

    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    uint32_t chip_id_offset = 0;
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() == tt::ClusterType::TG) {
        chip_id_offset = 4;
    }
    std::vector<chip_id_t> physical_chip_ids(num_devices);
    std::iota(physical_chip_ids.begin(), physical_chip_ids.end(), chip_id_offset);

    // find all neighbors of chip 0 (logical chip ids)
    std::unordered_map<tt_fabric::RoutingDirection, chip_id_t> chip_0_neigbors;
    std::optional<tt_fabric::RoutingDirection> chip_1_direction = std::nullopt;
    for (const auto& direction : routing_directions) {
        auto neighbors = control_plane.get_intra_chip_neighbors(FabricNodeId(mesh_id, 0), direction);
        if (neighbors.empty()) {
            continue;
        }
        // assuming same neighbor per direction
        chip_0_neigbors[direction] = neighbors[0];
        if (neighbors[0] == 1) {
            chip_1_direction = direction;
        }
    }

    if (chip_0_neigbors.size() > 2) {
        log_fatal(
            LogTest, "Expected 2 or less than 2 neigbors for a corner chip, but found {}", chip_0_neigbors.size());
        throw std::runtime_error("Unexpected number of neigbors for corner chip");
    }

    if (!chip_1_direction.has_value()) {
        throw std::runtime_error("Logical chip 1 is not a neighbor of logical chip 0");
    }

    int row_offset = 0, col_offset = 0;

    // determine the row and col offset which will be used while filling up the chip matrix
    switch (chip_1_direction.value()) {
        case tt_fabric::RoutingDirection::N: col_offset = -1; break;
        case tt_fabric::RoutingDirection::S: col_offset = 1; break;
        case tt_fabric::RoutingDirection::E: row_offset = 1; break;
        case tt_fabric::RoutingDirection::W: row_offset = -1; break;
        default: throw std::runtime_error("Unexpected direction");
    }

    if (chip_0_neigbors.size() == 2) {
        // find the other neighbor chip and direction
        tt_fabric::RoutingDirection other_chip_dir{};
        chip_id_t other_chip{};
        for (const auto& [dir, chip] : chip_0_neigbors) {
            if (chip != 0) {
                other_chip_dir = dir;
                other_chip = chip;
            }
        }

        switch (other_chip_dir) {
            case tt_fabric::RoutingDirection::N: col_offset = 0 - other_chip; break;  // chip 0 could be in E/W
            case tt_fabric::RoutingDirection::S: col_offset = other_chip; break;      // chip 0 could be in E/W
            case tt_fabric::RoutingDirection::E: row_offset = other_chip; break;      // chip 0 could be in N/S
            case tt_fabric::RoutingDirection::W: row_offset = 0 - other_chip; break;  // chip 0 could be in N/S
            default: throw std::runtime_error("Unexpected direction");
        }

        if (row_offset == 0 || col_offset == 0) {
            throw std::runtime_error("Unexpected error while setting up neighbor map");
        }
    }

    uint32_t num_rows = 0;
    uint32_t num_cols = 0;
    if (std::abs(row_offset) > 1 || col_offset == 0) {
        num_rows = std::abs(row_offset);
        num_cols = physical_chip_ids.size() / num_rows;
    } else if (std::abs(col_offset) > 1 || row_offset == 0) {
        num_cols = std::abs(col_offset);
        num_rows = physical_chip_ids.size() / num_cols;
    }

    if (num_rows == 0 || num_cols == 0) {
        throw std::runtime_error("Unable to determine number of rows or columns while setting up neighbor map");
    }

    // determine the chip for the NW corner of the matrix
    chip_id_t start_logical_chip_id = (num_rows - 1) * (col_offset < 0 ? std::abs(col_offset) : 0) +
                                      (num_cols - 1) * (row_offset < 0 ? std::abs(row_offset) : 0);

    // populate the physical chip matrix
    std::vector<std::vector<chip_id_t>> physical_chip_matrix(num_rows, std::vector<chip_id_t>(num_cols));
    for (uint32_t i = 0; i < num_rows; i++) {
        chip_id_t logical_chip_id = start_logical_chip_id;
        for (uint32_t j = 0; j < num_cols; j++) {
            if (logical_chip_id > physical_chip_ids.size()) {
                throw std::runtime_error("Failed to setup neighbor map, logical chip id exceeding bounds");
            }
            chip_id_t phys_chip_id =
                control_plane.get_physical_chip_id_from_fabric_node_id(FabricNodeId(mesh_id, logical_chip_id));
            physical_chip_matrix[i][j] = phys_chip_id;
            logical_chip_id += row_offset;
        }
        start_logical_chip_id += col_offset;
    }

    // try to get the chips from the rows first
    std::vector<chip_id_t> chip_seq;
    if (num_seq_chips <= num_cols) {
        for (auto i = 0; i < num_seq_chips; i++) {
            chip_seq.push_back(physical_chip_matrix[0][i]);
        }
    } else if (num_seq_chips <= num_rows) {
        for (auto i = 0; i < num_seq_chips; i++) {
            chip_seq.push_back(physical_chip_matrix[i][0]);
        }
    }

    return chip_seq;
}

uint32_t get_sender_id(CoreCoord logical_core) { return logical_core.x << 16 || logical_core.y; }

void create_kernel(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program_handle,
    const std::string& kernel_src,
    const CoreCoord& logical_core,
    const std::vector<uint32_t>& ct_args,
    const std::vector<uint32_t>& rt_args,
    const std::vector<std::pair<size_t, size_t>>& addresses_to_clear) {
    auto kernel_handle = tt::tt_metal::CreateKernel(
        program_handle,
        kernel_src,
        {logical_core},
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = ct_args,
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});
    tt::tt_metal::SetRuntimeArgs(program_handle, kernel_handle, logical_core, rt_args);

    for (const auto& [start_address, num_bytes] : addresses_to_clear) {
        std::vector<uint32_t> zero_vec((num_bytes / sizeof(uint32_t)), 0);
        tt::tt_metal::detail::WriteToDeviceL1(device, logical_core, start_address, zero_vec);
    }
}

void create_mux_kernel(
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    const CoreCoord& mux_logical_core,
    tt::tt_metal::IDevice* device,
    tt::tt_metal::IDevice* dest_device,
    tt::tt_metal::Program& program_handle) {
    std::vector<uint32_t> mux_ct_args = mux_kernel_config.get_fabric_mux_compile_time_args();
    std::vector<uint32_t> mux_rt_args = {};
    append_fabric_connection_rt_args(
        device->id(),
        dest_device->id(),
        0 /* link_idx (routing plane) */,
        program_handle,
        {mux_logical_core},
        mux_rt_args);

    std::vector<std::pair<size_t, size_t>> addresses_to_clear = {
        std::make_pair(mux_kernel_config.get_start_address_to_clear(), mux_kernel_config.get_num_bytes_to_clear())};
    create_kernel(
        device, program_handle, mux_kernel_src, mux_logical_core, mux_ct_args, mux_rt_args, addresses_to_clear);
}

void create_worker_kernel(
    const TestConfig& test_config,
    const WorkerTestConfig& worker_test_config,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    const CoreCoord& mux_virtual_core,
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program_handle) {
    auto worker_memory_map = worker_test_config.memory_map;
    CoreCoord worker_logical_core = worker_test_config.worker_logical_core;
    auto channel_type = worker_test_config.channel_type;
    auto worker_id = worker_test_config.worker_id;

    std::vector<uint32_t> worker_ct_args = {
        mux_virtual_core.x,
        mux_virtual_core.y,
        worker_test_config.num_buffers,
        worker_test_config.buffer_size_bytes,
        mux_kernel_config.get_channel_base_address(channel_type, worker_id),
        mux_kernel_config.get_connection_info_address(channel_type, worker_id),
        mux_kernel_config.get_connection_handshake_address(channel_type, worker_id),
        mux_kernel_config.get_flow_control_address(channel_type, worker_id),
        mux_kernel_config.get_buffer_index_address(channel_type, worker_id),
        mux_kernel_config.get_status_address(),
        mux_kernel_config.get_channel_credits_stream_id(channel_type, worker_id)};

    // virtual coordinates will be the same for the receiver device
    // hence, we can use the noc encoding derived using current device
    CoreCoord worker_virtual_core = device->worker_core_from_logical_core(worker_logical_core);
    uint32_t receiver_noc_xy_encoding =
        tt_metal::MetalContext::instance().hal().noc_xy_encoding(worker_virtual_core.x, worker_virtual_core.y);

    std::vector<uint32_t> worker_rt_args = {
        test_config.num_open_close_iters,
        test_config.num_packets,
        test_config.num_credits,
        test_config.packet_payload_size_bytes,
        test_config.time_seed,
        test_config.num_return_credits_per_packet,
        test_results_size_bytes,
        worker_memory_map->test_results_address,
        worker_memory_map->local_mux_status_address,
        worker_memory_map->local_flow_control_address,
        worker_memory_map->local_teardown_address,
        worker_memory_map->local_buffer_index_address,
        worker_memory_map->base_l1_target_address,
        worker_memory_map->credit_handshake_address,
        worker_memory_map->packet_header_buffer_address,
        worker_memory_map->payload_buffer_address,
        worker_test_config.num_hops,
        get_sender_id(worker_logical_core),
        receiver_noc_xy_encoding};

    std::vector<std::pair<size_t, size_t>> addresses_to_clear = {
        std::make_pair(worker_memory_map->local_flow_control_address, noc_address_padding_bytes),
        std::make_pair(worker_memory_map->local_teardown_address, noc_address_padding_bytes),
        std::make_pair(worker_memory_map->local_buffer_index_address, noc_address_padding_bytes)};
    create_kernel(
        device,
        program_handle,
        std::string(worker_test_config.kernel_src),
        worker_logical_core,
        worker_ct_args,
        worker_rt_args,
        addresses_to_clear);
}

void run_mux_test_variant(FabricMuxFixture* fixture, TestConfig test_config) {
    auto num_devices = test_config.num_devices;
    auto chip_seq = get_physical_chip_sequence(num_devices);
    if (chip_seq.empty()) {
        GTEST_SKIP() << "Not enough devices available";
    }
    log_info(LogTest, "Devices: {}", chip_seq);

    std::vector<tt::tt_metal::IDevice*> devices;
    for (const auto& chip_id : chip_seq) {
        devices.push_back(DevicePool::instance().get_active_device(chip_id));
    }

    // dest devices for mux kernel to attach to the fabric routers
    std::unordered_map<tt::tt_metal::IDevice*, tt::tt_metal::IDevice*> dest_devices;
    dest_devices[devices[0]] = devices[1];
    for (auto i = 1; i < num_devices; i++) {
        dest_devices[devices[i]] = devices[i - 1];
    }

    // TODO: assert on the number of minimum senders/receivers for the device seq
    const uint8_t num_senders = test_config.num_sender_clients;
    const uint8_t num_receivers = test_config.num_sender_clients;
    if (num_senders < num_devices - 1) {
        GTEST_SKIP() << "Not enough senders/receivers for this configuration";
    }

    // [device] -> {(logical_core, hops)}
    // keeps track of which logical cores will talk to each other and are how many hops away
    std::unordered_map<tt::tt_metal::IDevice*, std::vector<std::pair<CoreCoord, uint8_t>>> device_senders_map;
    std::unordered_map<tt::tt_metal::IDevice*, std::vector<std::pair<CoreCoord, uint8_t>>> device_receivers_map;

    // number of senders/receivers starting 2nd device in the sequence if uniformly distributed
    uint8_t num_senders_per_chip = num_senders / (num_devices - 1);
    uint8_t num_receivers_per_chip = num_receivers / (num_devices - 1);

    // if the sender-reciever split is uniform, i.e., same number of senders and receivers on a device,
    // the number of full size and header only channels
    // will be the same across devices, else the deivces (following the 1st one
    // in the sequence) will have unequal number of full size and header only channels
    int8_t offset = 0;
    if (!test_config.uniform_sender_receiver_split) {
        // for every pair of devices, flip-flop b/w the assigned workers, i.e, if uniform distribution is 2
        // chip 1 gets 3 (2 + 1) and chip 2 gets 1 (2 - 1)
        offset = 1;
    }

    std::vector<CoreCoord> worker_logical_cores;
    auto grid_size = devices[0]->compute_with_storage_grid_size();
    for (auto i = 0; i < grid_size.x; i++) {
        for (auto j = 0; j < grid_size.y; j++) {
            worker_logical_cores.push_back(CoreCoord({i, j}));
        }
    }

    auto assign_worker_cores =
        [&](tt::tt_metal::IDevice* device, uint8_t num_workers, uint8_t num_hops, bool is_sender) {
            for (uint8_t i = 0; i < num_workers; i++) {
                auto core_and_hops = std::make_pair(worker_logical_cores.back(), num_hops);
                worker_logical_cores.pop_back();
                if (is_sender) {
                    device_senders_map[device].push_back(core_and_hops);
                    device_receivers_map[devices.front()].push_back(core_and_hops);
                } else {
                    device_receivers_map[device].push_back(core_and_hops);
                    device_senders_map[devices.front()].push_back(core_and_hops);
                }
            }
        };

    uint8_t num_assigned_senders = 0;
    uint8_t num_assigned_receivers = 0;
    // each of the senders starting from the 2nd device in the seq will send packets to the 1st device
    // each of the receivers starting from the 2nd device in the seq will receive packets from the 1st device
    for (auto i = 1; i < num_devices - 1; i++) {
        auto num_local_senders = num_senders_per_chip + offset;
        assign_worker_cores(devices[i], num_local_senders, i, true /* is_sender */);
        num_assigned_senders += num_local_senders;

        offset *= -1;
        auto num_local_receivers = num_receivers_per_chip + offset;
        assign_worker_cores(devices[i], num_local_senders, i, false /* is_sender */);
        num_assigned_receivers += num_local_receivers;
    }

    // assign all the remaining senders/receivers to the last device
    assign_worker_cores(devices.back(), num_senders - num_assigned_senders, num_devices - 1, true /* is_sender */);
    assign_worker_cores(devices.back(), num_receivers - num_assigned_receivers, num_devices - 1, false /* is_sender */);

    const uint32_t l1_unreserved_base_address =
        devices[0]->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

    const size_t mux_base_l1_address = l1_unreserved_base_address;
    auto worker_memory_map = create_worker_memory_map(l1_unreserved_base_address);

    std::vector<tt::tt_metal::Program> program_handles(devices.size());
    std::vector<size_t> mux_termination_signal_addresses;

    size_t buffer_size_bytes_full_size_channel =
        sizeof(tt::tt_fabric::PacketHeader) + test_config.packet_payload_size_bytes;
    size_t buffer_size_bytes_header_only_channel = sizeof(tt::tt_fabric::PacketHeader);

    for (auto i = 0; i < devices.size(); i++) {
        program_handles[i] = tt_metal::CreateProgram();
        // use logical core (0,0) for the mux kernel
        CoreCoord mux_logical_core = worker_logical_cores[0];
        CoreCoord mux_virtual_core = devices[i]->worker_core_from_logical_core(mux_logical_core);

        auto num_full_size_channels = device_senders_map[devices[i]].size();
        auto num_header_only_channels = device_receivers_map[devices[i]].size();

        auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
            num_full_size_channels,
            num_header_only_channels,
            test_config.num_buffers_full_size_channel,
            test_config.num_buffers_header_only_channel,
            buffer_size_bytes_full_size_channel,
            mux_base_l1_address);
        if (test_config.num_full_size_channel_iters > 1) {
            mux_kernel_config.set_num_full_size_channel_iters(test_config.num_full_size_channel_iters);
        }

        mux_termination_signal_addresses.push_back(mux_kernel_config.get_termination_signal_address());

        create_mux_kernel(
            mux_kernel_config, mux_logical_core, devices[i], dest_devices[devices[i]], program_handles[i]);

        uint32_t sender_id = 0;
        for (const auto& [sender_logical_core, num_hops] : device_senders_map[devices[i]]) {
            WorkerTestConfig sender_config = {
                .memory_map = &worker_memory_map,
                .worker_logical_core = sender_logical_core,
                .worker_id = sender_id,
                .channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                .num_buffers = test_config.num_buffers_full_size_channel,
                .buffer_size_bytes = buffer_size_bytes_full_size_channel,
                .num_hops = num_hops,
                .kernel_src = sender_kernel_src};
            create_worker_kernel(
                test_config, sender_config, mux_kernel_config, mux_virtual_core, devices[i], program_handles[i]);
            sender_id++;
        }

        uint32_t receiver_id = 0;
        for (const auto& [receiver_logical_core, num_hops] : device_receivers_map[devices[i]]) {
            WorkerTestConfig receiver_config = {
                .memory_map = &worker_memory_map,
                .worker_logical_core = receiver_logical_core,
                .worker_id = receiver_id,
                .channel_type = tt::tt_fabric::FabricMuxChannelType::HEADER_ONLY_CHANNEL,
                .num_buffers = test_config.num_buffers_header_only_channel,
                .buffer_size_bytes = buffer_size_bytes_header_only_channel,
                .num_hops = num_hops,
                .kernel_src = receiver_kernel_src};
            create_worker_kernel(
                test_config, receiver_config, mux_kernel_config, mux_virtual_core, devices[i], program_handles[i]);
            receiver_id++;
        }
    }

    log_info(LogTest, "Running programs");
    for (auto i = 0; i < devices.size(); i++) {
        fixture->RunProgramNonblocking(devices[i], program_handles[i]);
    }

    auto wait_for_worker_completion = [&](tt::tt_metal::IDevice* device, const CoreCoord& core) {
        std::vector<uint32_t> worker_status(1, 0);
        while ((worker_status[0] & 0xFFFF) == 0) {
            tt_metal::detail::ReadFromDeviceL1(device, core, worker_memory_map.test_results_address, 4, worker_status);
        }
    };

    log_info(LogTest, "Waiting for senders to complete");
    for (auto i = 0; i < devices.size(); i++) {
        for (const auto& [core, _] : device_senders_map[devices[i]]) {
            wait_for_worker_completion(devices[i], core);
        }
    }

    log_info(LogTest, "Senders done, waiting for receivers to complete");
    for (auto i = 0; i < devices.size(); i++) {
        for (const auto& [core, _] : device_receivers_map[devices[i]]) {
            wait_for_worker_completion(devices[i], core);
        }
    }

    log_info(LogTest, "Receivers done, terminating mux kernel");
    std::vector<uint32_t> mux_termination_signal(1, tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE);
    for (auto i = 0; i < devices.size(); i++) {
        tt::tt_metal::detail::WriteToDeviceL1(
            devices[i], worker_logical_cores[0], mux_termination_signal_addresses[i], mux_termination_signal);
    }

    log_info(LogTest, "Waiting for programs");
    for (auto i = 0; i < devices.size(); i++) {
        fixture->WaitForSingleProgramDone(devices[i], program_handles[i]);
    }

    auto validate_worker_results = [&](tt::tt_metal::IDevice* device, const CoreCoord& core) {
        std::vector<uint32_t> worker_status;
        tt_metal::detail::ReadFromDeviceL1(
            device, core, worker_memory_map.test_results_address, test_results_size_bytes, worker_status);
        EXPECT_EQ(worker_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
    };

    log_info(LogTest, "Programs done, validating results");
    for (auto i = 0; i < devices.size(); i++) {
        for (const auto& [core, _] : device_senders_map[devices[i]]) {
            validate_worker_results(devices[i], core);
        }

        for (const auto& [core, _] : device_receivers_map[devices[i]]) {
            validate_worker_results(devices[i], core);
        }
    }
}

TEST_F(FabricMuxFixture, TestFabricMuxTwoChipVariant1) {
    TestConfig test_config = {
        .num_devices = 2,
        .num_sender_clients = 1,
        .num_packets = 1000,
        .num_credits = 1,
        .num_return_credits_per_packet = 1,
        .packet_payload_size_bytes = 4096,
        .time_seed = std::chrono::system_clock::now().time_since_epoch().count(),
        .num_buffers_full_size_channel = 1,
        .num_buffers_header_only_channel = 1,
        .uniform_sender_receiver_split = true,
        .num_open_close_iters = 1,
        .num_full_size_channel_iters = 1,
    };
    run_mux_test_variant(this, test_config);
}

TEST_F(FabricMuxFixture, TestFabricMuxTwoChipVariant2) {
    TestConfig test_config = {
        .num_devices = 2,
        .num_sender_clients = 2,
        .num_packets = 1000,
        .num_credits = 1,
        .num_return_credits_per_packet = 1,
        .packet_payload_size_bytes = 4096,
        .time_seed = std::chrono::system_clock::now().time_since_epoch().count(),
        .num_buffers_full_size_channel = 4,
        .num_buffers_header_only_channel = 4,
        .uniform_sender_receiver_split = true,
        .num_open_close_iters = 1,
        .num_full_size_channel_iters = 1,
    };
    run_mux_test_variant(this, test_config);
}

TEST_F(FabricMuxFixture, TestFabricMuxTwoChipVariant3) {
    TestConfig test_config = {
        .num_devices = 2,
        .num_sender_clients = 8,
        .num_packets = 5000,
        .num_credits = 16,
        .num_return_credits_per_packet = 1,
        .packet_payload_size_bytes = 4096,
        .time_seed = std::chrono::system_clock::now().time_since_epoch().count(),
        .num_buffers_full_size_channel = 8,
        .num_buffers_header_only_channel = 8,
        .uniform_sender_receiver_split = true,
        .num_open_close_iters = 1,
        .num_full_size_channel_iters = 1,
    };
    run_mux_test_variant(this, test_config);
}

TEST_F(FabricMuxFixture, TestFabricMuxTwoChipVariant4) {
    TestConfig test_config = {
        .num_devices = 2,
        .num_sender_clients = 8,
        .num_packets = 50000,
        .num_credits = 16,
        .num_return_credits_per_packet = 8,
        .packet_payload_size_bytes = 4096,
        .time_seed = std::chrono::system_clock::now().time_since_epoch().count(),
        .num_buffers_full_size_channel = 8,
        .num_buffers_header_only_channel = 8,
        .uniform_sender_receiver_split = true,
        .num_open_close_iters = 1,
        .num_full_size_channel_iters = 1,
    };
    run_mux_test_variant(this, test_config);
}

TEST_F(FabricMuxFixture, TestFabricMuxThreeChipVariant) {
    TestConfig test_config = {
        .num_devices = 3,
        .num_sender_clients = 8,
        .num_packets = 5000,
        .num_credits = 16,
        .num_return_credits_per_packet = 8,
        .packet_payload_size_bytes = 4096,
        .time_seed = std::chrono::system_clock::now().time_since_epoch().count(),
        .num_buffers_full_size_channel = 8,
        .num_buffers_header_only_channel = 8,
        .uniform_sender_receiver_split = true,
        .num_open_close_iters = 1,
        .num_full_size_channel_iters = 1,
    };
    run_mux_test_variant(this, test_config);
}

TEST_F(FabricMuxFixture, TestFabricMuxFourChipVariant1) {
    TestConfig test_config = {
        .num_devices = 4,
        .num_sender_clients = 8,
        .num_packets = 5000,
        .num_credits = 16,
        .num_return_credits_per_packet = 8,
        .packet_payload_size_bytes = 4096,
        .time_seed = std::chrono::system_clock::now().time_since_epoch().count(),
        .num_buffers_full_size_channel = 8,
        .num_buffers_header_only_channel = 8,
        .uniform_sender_receiver_split = true,
        .num_open_close_iters = 1,
        .num_full_size_channel_iters = 1,
    };
    run_mux_test_variant(this, test_config);
}

TEST_F(FabricMuxFixture, TestFabricMuxFourChipVariant2) {
    TestConfig test_config = {
        .num_devices = 4,
        .num_sender_clients = 8,
        .num_packets = 5000,
        .num_credits = 16,
        .num_return_credits_per_packet = 8,
        .packet_payload_size_bytes = 4096,
        .time_seed = std::chrono::system_clock::now().time_since_epoch().count(),
        .num_buffers_full_size_channel = 8,
        .num_buffers_header_only_channel = 8,
        .uniform_sender_receiver_split = false,
        .num_open_close_iters = 1,
        .num_full_size_channel_iters = 1,
    };
    run_mux_test_variant(this, test_config);
}

TEST_F(FabricMuxFixture, TestFabricMuxFiveChipVariant) {
    TestConfig test_config = {
        .num_devices = 5,
        .num_sender_clients = 8,
        .num_packets = 5000,
        .num_credits = 16,
        .num_return_credits_per_packet = 8,
        .packet_payload_size_bytes = 4096,
        .time_seed = std::chrono::system_clock::now().time_since_epoch().count(),
        .num_buffers_full_size_channel = 8,
        .num_buffers_header_only_channel = 8,
        .uniform_sender_receiver_split = true,
        .num_open_close_iters = 1,
        .num_full_size_channel_iters = 1,
    };
    run_mux_test_variant(this, test_config);
}

TEST_F(FabricMuxFixture, TestFabricMuxSixChipVariant) {
    TestConfig test_config = {
        .num_devices = 6,
        .num_sender_clients = 16,
        .num_packets = 5000,
        .num_credits = 16,
        .num_return_credits_per_packet = 8,
        .packet_payload_size_bytes = 4096,
        .time_seed = std::chrono::system_clock::now().time_since_epoch().count(),
        .num_buffers_full_size_channel = 4,
        .num_buffers_header_only_channel = 4,
        .uniform_sender_receiver_split = true,
        .num_open_close_iters = 1,
        .num_full_size_channel_iters = 2,
    };
    run_mux_test_variant(this, test_config);
}

TEST_F(FabricMuxFixture, TestFabricMuxSevenChipVariant) {
    TestConfig test_config = {
        .num_devices = 7,
        .num_sender_clients = 20,
        .num_packets = 5000,
        .num_credits = 16,
        .num_return_credits_per_packet = 8,
        .packet_payload_size_bytes = 4096,
        .time_seed = std::chrono::system_clock::now().time_since_epoch().count(),
        .num_buffers_full_size_channel = 4,
        .num_buffers_header_only_channel = 4,
        .uniform_sender_receiver_split = true,
        .num_open_close_iters = 1,
        .num_full_size_channel_iters = 3,
    };
    run_mux_test_variant(this, test_config);
}

TEST_F(FabricMuxFixture, TestFabricMuxEightChipVariant) {
    TestConfig test_config = {
        .num_devices = 8,
        .num_sender_clients = 20,
        .num_packets = 5000,
        .num_credits = 16,
        .num_return_credits_per_packet = 8,
        .packet_payload_size_bytes = 4096,
        .time_seed = std::chrono::system_clock::now().time_since_epoch().count(),
        .num_buffers_full_size_channel = 4,
        .num_buffers_header_only_channel = 4,
        .uniform_sender_receiver_split = false,
        .num_open_close_iters = 1,
        .num_full_size_channel_iters = 3,
    };
    run_mux_test_variant(this, test_config);
}

TEST_F(FabricMuxFixture, TestFabricMuxStressOpenClose) {
    TestConfig test_config = {
        .num_devices = 2,  // running on 2 devices will allow to test on all types of multi-chip systems
        .num_sender_clients = 8,
        .num_packets = 100,
        .num_credits = 16,
        .num_return_credits_per_packet = 8,
        .packet_payload_size_bytes = 4096,
        .time_seed = std::chrono::system_clock::now().time_since_epoch().count(),
        .num_buffers_full_size_channel = 8,
        .num_buffers_header_only_channel = 8,
        .uniform_sender_receiver_split = false,
        .num_open_close_iters = 10000,
        .num_full_size_channel_iters = 1,
    };
    run_mux_test_variant(this, test_config);
}

TEST_F(FabricMuxFixture, TestFabricMuxNumFullSizeChannelIters) {
    TestConfig test_config = {
        .num_devices = 2,  // running on 2 devices will allow to test on all types of multi-chip systems
        .num_sender_clients = 8,
        .num_packets = 100000,
        .num_credits = 16,
        .num_return_credits_per_packet = 8,
        .packet_payload_size_bytes = 4096,
        .time_seed = std::chrono::system_clock::now().time_since_epoch().count(),
        .num_buffers_full_size_channel = 8,
        .num_buffers_header_only_channel = 8,
        .uniform_sender_receiver_split = true,
        .num_open_close_iters = 1,
        .num_full_size_channel_iters = 2,
    };
    run_mux_test_variant(this, test_config);
}

}  // namespace fabric_mux_tests
}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
