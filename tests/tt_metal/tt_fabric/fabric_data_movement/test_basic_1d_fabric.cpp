// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <tt-metalium/fabric_host_interface.h>
#include <array>
#include <cstddef>
#include <map>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include "fabric_fixture.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/fabric.hpp>
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "umd/device/tt_core_coordinates.h"

namespace tt::tt_fabric {
namespace fabric_router_tests {

bool find_device_with_neighbor_in_direction(
    BaseFabricFixture* fixture,
    std::pair<mesh_id_t, chip_id_t>& src_mesh_chip_id,
    std::pair<mesh_id_t, chip_id_t>& dst_mesh_chip_id,
    chip_id_t& src_physical_device_id,
    chip_id_t& dst_physical_device_id,
    RoutingDirection direction) {
    auto* control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();
    auto devices = fixture->get_devices();
    for (auto* device : devices) {
        src_mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());

        // Get neighbours within a mesh in the given direction
        auto neighbors =
            control_plane->get_intra_chip_neighbors(src_mesh_chip_id.first, src_mesh_chip_id.second, direction);
        if (neighbors.size() > 0) {
            src_physical_device_id = device->id();
            dst_mesh_chip_id = {src_mesh_chip_id.first, neighbors[0]};
            dst_physical_device_id = control_plane->get_physical_chip_id_from_mesh_chip_id(dst_mesh_chip_id);
            return true;
        }
    }
    return false;
}

// Find a device with enough neighbours in the specified direction
bool find_device_with_neighbor_in_multi_direction(
    BaseFabricFixture* fixture,
    std::pair<mesh_id_t, chip_id_t>& src_mesh_chip_id,
    std::unordered_map<RoutingDirection, std::vector<std::pair<mesh_id_t, chip_id_t>>>& dst_mesh_chip_ids_by_dir,
    chip_id_t& src_physical_device_id,
    std::unordered_map<RoutingDirection, std::vector<chip_id_t>>& dst_physical_device_ids_by_dir,
    const std::unordered_map<RoutingDirection, uint32_t>& mcast_hops) {
    auto control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();

    auto devices = fixture->get_devices();
    // Find a device with enough neighbours in the specified direction
    bool connection_found = false;
    for (auto* device : devices) {
        src_mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());
        std::unordered_map<RoutingDirection, std::vector<std::pair<mesh_id_t, chip_id_t>>>
            temp_end_mesh_chip_ids_by_dir;
        std::unordered_map<RoutingDirection, std::vector<chip_id_t>> temp_physical_end_device_ids_by_dir;
        connection_found = true;
        for (auto [routing_direction, num_hops] : mcast_hops) {
            bool direction_found = true;
            auto& temp_end_mesh_chip_ids = temp_end_mesh_chip_ids_by_dir[routing_direction];
            auto& temp_physical_end_device_ids = temp_physical_end_device_ids_by_dir[routing_direction];
            uint32_t curr_mesh_id = src_mesh_chip_id.first;
            uint32_t curr_chip_id = src_mesh_chip_id.second;
            for (uint32_t i = 0; i < num_hops; i++) {
                auto neighbors = control_plane->get_intra_chip_neighbors(curr_mesh_id, curr_chip_id, routing_direction);
                if (neighbors.size() > 0) {
                    temp_end_mesh_chip_ids.emplace_back(curr_mesh_id, neighbors[0]);
                    temp_physical_end_device_ids.push_back(
                        control_plane->get_physical_chip_id_from_mesh_chip_id(temp_end_mesh_chip_ids.back()));
                    curr_mesh_id = temp_end_mesh_chip_ids.back().first;
                    curr_chip_id = temp_end_mesh_chip_ids.back().second;
                } else {
                    direction_found = false;
                    break;
                }
            }
            if (!direction_found) {
                connection_found = false;
                break;
            }
        }
        if (connection_found) {
            src_physical_device_id = device->id();
            dst_mesh_chip_ids_by_dir = std::move(temp_end_mesh_chip_ids_by_dir);
            dst_physical_device_ids_by_dir = std::move(temp_physical_end_device_ids_by_dir);
            break;
        }
    }
    return connection_found;
}

void RunTestUnicastRaw(BaseFabricFixture* fixture, uint32_t num_hops, RoutingDirection direction) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};

    auto* control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();

    std::pair<mesh_id_t, chip_id_t> src_mesh_chip_id;
    std::pair<mesh_id_t, chip_id_t> dst_mesh_chip_id;
    chip_id_t not_used_1;
    chip_id_t not_used_2;
    // Find a device num_hops away in specified direction.
    std::unordered_map<RoutingDirection, uint32_t> fabric_hops;
    std::unordered_map<RoutingDirection, std::vector<std::pair<mesh_id_t, chip_id_t>>> end_mesh_chip_ids_by_dir;
    chip_id_t src_physical_device_id;
    std::unordered_map<RoutingDirection, std::vector<chip_id_t>> physical_end_device_ids_by_dir;
    fabric_hops[direction] = num_hops;
    // Find a device with enough neighbours in the specified directions
    if (!find_device_with_neighbor_in_multi_direction(
            fixture,
            src_mesh_chip_id,
            end_mesh_chip_ids_by_dir,
            src_physical_device_id,
            physical_end_device_ids_by_dir,
            fabric_hops)) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    chip_id_t dst_physical_device_id = physical_end_device_ids_by_dir[direction][num_hops - 1];
    dst_mesh_chip_id = end_mesh_chip_ids_by_dir[direction][num_hops - 1];

    tt::log_info(tt::LogTest, "mesh/chip ids {}", end_mesh_chip_ids_by_dir[direction]);
    tt::log_info(tt::LogTest, "physical device ids {}", physical_end_device_ids_by_dir[direction]);
    tt::log_info(tt::LogTest, "Src MeshId {} ChipId {}", src_mesh_chip_id.first, src_mesh_chip_id.second);
    tt::log_info(tt::LogTest, "Dst MeshId {} ChipId {}", dst_mesh_chip_id.first, dst_mesh_chip_id.second);
    tt::log_info(tt::LogTest, "Dst Device is {} hops in direction: {}", num_hops, direction);

    // get a port to connect to
    std::set<chan_id_t> eth_chans = control_plane->get_active_fabric_eth_channels_in_direction(
        src_mesh_chip_id.first, src_mesh_chip_id.second, direction);
    if (eth_chans.size() == 0) {
        GTEST_SKIP() << "No active eth chans to connect to";
    }

    // Pick a port from end of the list. On T3K, there are missimg routing planes due to FD tunneling
    auto edm_port = *std::prev(eth_chans.end());

    auto edm_direction =
        control_plane->get_eth_chan_direction(src_mesh_chip_id.first, src_mesh_chip_id.second, edm_port);
    CoreCoord edm_eth_core = tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_eth_core_from_channel(
        src_physical_device_id, edm_port);

    auto* sender_device = DevicePool::instance().get_active_device(src_physical_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(dst_physical_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    auto receiver_noc_encoding =
        tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    const auto edm_config = get_tt_fabric_config();
    uint32_t is_2d_fabric = edm_config.topology == Topology::Mesh;

    // test parameters
    uint32_t packet_header_address = 0x25000;
    uint32_t source_l1_buffer_address = 0x30000;
    uint32_t packet_payload_size_bytes = edm_config.topology == Topology::Mesh ? 2048 : 4096;
    uint32_t num_packets = 10;
    uint32_t test_results_address = 0x100000;
    uint32_t test_results_size_bytes = 128;
    uint32_t target_address = 0x30000;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    // common compile time args for sender and receiver
    std::vector<uint32_t> compile_time_args = {
        test_results_address,
        test_results_size_bytes,
        target_address,
        0 /* mcast_mode */,
        edm_config.topology == Topology::Mesh};

    std::map<string, string> defines = {};
    if (is_2d_fabric) {
        defines["FABRIC_2D"] = "";
    }

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_tx.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args,
            .defines = defines});

    auto mesh_shape = control_plane->get_physical_mesh_shape(src_mesh_chip_id.first);
    tt::log_info(tt::LogTest, "mesh dimensions {:x}", mesh_shape.dims());
    tt::log_info(tt::LogTest, "mesh dimension 0 {:x}", mesh_shape[0]);
    tt::log_info(tt::LogTest, "mesh dimension 1 {:x}", mesh_shape[1]);

    std::vector<uint32_t> sender_runtime_args = {
        packet_header_address,
        source_l1_buffer_address,
        packet_payload_size_bytes,
        num_packets,
        receiver_noc_encoding,
        time_seed,
        mesh_shape[1],
        src_mesh_chip_id.second,
        dst_mesh_chip_id.second,
        num_hops};

    // append the EDM connection rt args
    const auto sender_channel = edm_config.topology == Topology::Mesh ? edm_direction : 0;
    tt::tt_fabric::SenderWorkerAdapterSpec edm_connection = {
        .edm_noc_x = edm_eth_core.x,
        .edm_noc_y = edm_eth_core.y,
        .edm_buffer_base_addr = edm_config.sender_channels_base_address[sender_channel],
        .num_buffers_per_channel = edm_config.sender_channels_num_buffers[sender_channel],
        .edm_l1_sem_addr = edm_config.sender_channels_local_flow_control_semaphore_address[sender_channel],
        .edm_connection_handshake_addr = edm_config.sender_channels_connection_semaphore_address[sender_channel],
        .edm_worker_location_info_addr = edm_config.sender_channels_worker_conn_info_base_address[sender_channel],
        .buffer_size_bytes = edm_config.channel_buffer_size_bytes,
        .buffer_index_semaphore_id = edm_config.sender_channels_buffer_index_semaphore_address[sender_channel],
        .persistent_fabric = true,
        .edm_direction = edm_direction};

    auto worker_flow_control_semaphore_id = tt_metal::CreateSemaphore(sender_program, sender_logical_core, 0);
    auto worker_teardown_semaphore_id = tt_metal::CreateSemaphore(sender_program, sender_logical_core, 0);
    auto worker_buffer_index_semaphore_id = tt_metal::CreateSemaphore(sender_program, sender_logical_core, 0);

    append_worker_to_fabric_edm_sender_rt_args(
        edm_connection,
        worker_flow_control_semaphore_id,
        worker_teardown_semaphore_id,
        worker_buffer_index_semaphore_id,
        sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    std::vector<uint32_t> receiver_runtime_args = {packet_payload_size_bytes, num_packets, time_seed};

    // Create the receiver program for validation
    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    // Launch sender and receiver programs and wait for them to finish
    fixture->RunProgramNonblocking(receiver_device, receiver_program);
    fixture->RunProgramNonblocking(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(receiver_device, receiver_program);

    // Validate the status and packets processed by sender and receiver
    std::vector<uint32_t> sender_status;
    std::vector<uint32_t> receiver_status;

    tt_metal::detail::ReadFromDeviceL1(
        sender_device,
        sender_logical_core,
        test_results_address,
        test_results_size_bytes,
        sender_status,
        CoreType::WORKER);

    tt_metal::detail::ReadFromDeviceL1(
        receiver_device,
        receiver_logical_core,
        test_results_address,
        test_results_size_bytes,
        receiver_status,
        CoreType::WORKER);

    EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
    EXPECT_EQ(receiver_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);

    uint64_t sender_bytes =
        ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];
    uint64_t receiver_bytes =
        ((uint64_t)receiver_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | receiver_status[TT_FABRIC_WORD_CNT_INDEX];
    EXPECT_EQ(sender_bytes, receiver_bytes);
}

void RunTestUnicastConnAPI(BaseFabricFixture* fixture, uint32_t num_hops, RoutingDirection direction) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};

    auto* control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();

    std::pair<mesh_id_t, chip_id_t> src_mesh_chip_id;
    std::pair<mesh_id_t, chip_id_t> dst_mesh_chip_id;
    chip_id_t not_used_1;
    chip_id_t not_used_2;
    // Find a device with a neighbour in the East direction
    bool connection_found = find_device_with_neighbor_in_direction(
        fixture, src_mesh_chip_id, dst_mesh_chip_id, not_used_1, not_used_2, direction);
    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    tt::log_info(tt::LogTest, "Src MeshId {} ChipId {}", src_mesh_chip_id.first, src_mesh_chip_id.second);
    tt::log_info(tt::LogTest, "Dst MeshId {} ChipId {}", dst_mesh_chip_id.first, dst_mesh_chip_id.second);
    tt::log_info(tt::LogTest, "Dst Device is {} hops in direction: {}", num_hops, direction);

    chip_id_t src_physical_device_id = control_plane->get_physical_chip_id_from_mesh_chip_id(src_mesh_chip_id);
    chip_id_t dst_physical_device_id = control_plane->get_physical_chip_id_from_mesh_chip_id(dst_mesh_chip_id);

    auto* sender_device = DevicePool::instance().get_active_device(src_physical_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(dst_physical_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    auto receiver_noc_encoding =
        tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);
    const auto edm_config = get_tt_fabric_config();
    uint32_t is_2d_fabric = edm_config.topology == Topology::Mesh;

    // test parameters
    uint32_t packet_header_address = 0x25000;
    uint32_t source_l1_buffer_address = 0x30000;
    uint32_t packet_payload_size_bytes = edm_config.topology == Topology::Mesh ? 2048 : 4096;
    uint32_t num_packets = 10;
    uint32_t test_results_address = 0x100000;
    uint32_t test_results_size_bytes = 128;
    uint32_t target_address = 0x30000;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    // common compile time args for sender and receiver
    std::vector<uint32_t> compile_time_args = {
        test_results_address,
        test_results_size_bytes,
        target_address,
        0 /* mcast_mode */,
        edm_config.topology == Topology::Mesh};

    std::map<string, string> defines = {};
    if (is_2d_fabric) {
        defines["FABRIC_2D"] = "";
    }

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_tx.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args,
            .defines = defines});

    auto mesh_shape = control_plane->get_physical_mesh_shape(src_mesh_chip_id.first);
    tt::log_info(tt::LogTest, "mesh dimensions {:x}", mesh_shape.dims());
    tt::log_info(tt::LogTest, "mesh dimension 0 {:x}", mesh_shape[0]);
    tt::log_info(tt::LogTest, "mesh dimension 1 {:x}", mesh_shape[1]);

    std::vector<uint32_t> sender_runtime_args = {
        packet_header_address,
        source_l1_buffer_address,
        packet_payload_size_bytes,
        num_packets,
        receiver_noc_encoding,
        time_seed,
        mesh_shape[1],
        src_mesh_chip_id.second,
        dst_mesh_chip_id.second,
        num_hops};

    // append the EDM connection rt args
    append_fabric_connection_rt_args(
        src_physical_device_id, dst_physical_device_id, 0, sender_program, {sender_logical_core}, sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    std::vector<uint32_t> receiver_runtime_args = {packet_payload_size_bytes, num_packets, time_seed};

    // Create the receiver program for validation
    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    // Launch sender and receiver programs and wait for them to finish
    fixture->RunProgramNonblocking(receiver_device, receiver_program);
    fixture->RunProgramNonblocking(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(receiver_device, receiver_program);

    // Validate the status and packets processed by sender and receiver
    std::vector<uint32_t> sender_status;
    std::vector<uint32_t> receiver_status;

    tt_metal::detail::ReadFromDeviceL1(
        sender_device,
        sender_logical_core,
        test_results_address,
        test_results_size_bytes,
        sender_status,
        CoreType::WORKER);

    tt_metal::detail::ReadFromDeviceL1(
        receiver_device,
        receiver_logical_core,
        test_results_address,
        test_results_size_bytes,
        receiver_status,
        CoreType::WORKER);

    EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
    EXPECT_EQ(receiver_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);

    uint64_t sender_bytes =
        ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];
    uint64_t receiver_bytes =
        ((uint64_t)receiver_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | receiver_status[TT_FABRIC_WORD_CNT_INDEX];
    EXPECT_EQ(sender_bytes, receiver_bytes);
}

void RunTestMCastConnAPI(BaseFabricFixture* fixture) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};

    auto control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();

    // use control plane to find a mesh with 3 devices
    auto user_meshes = control_plane->get_user_physical_mesh_ids();
    std::optional<mesh_id_t> mesh_id;
    for (const auto& mesh : user_meshes) {
        auto mesh_shape = control_plane->get_physical_mesh_shape(mesh);
        if (mesh_shape.mesh_size() >= 3) {
            mesh_id = mesh;
            break;
        }
    }
    if (!mesh_id.has_value()) {
        GTEST_SKIP() << "No mesh found for 3 chip mcast test";
    }

    chip_id_t src_chip_id = 1;
    chip_id_t left_chip_id = 0;
    chip_id_t right_chip_id = 2;
    // for this test, logical chip id 1 is the sender, 0 is the left receiver and 1 is the right receiver
    auto src_phys_chip_id =
        control_plane->get_physical_chip_id_from_mesh_chip_id(std::make_pair(mesh_id.value(), src_chip_id));
    auto left_recv_phys_chip_id =
        control_plane->get_physical_chip_id_from_mesh_chip_id(std::make_pair(mesh_id.value(), left_chip_id));
    auto right_recv_phys_chip_id =
        control_plane->get_physical_chip_id_from_mesh_chip_id(std::make_pair(mesh_id.value(), right_chip_id));

    auto* sender_device = DevicePool::instance().get_active_device(src_phys_chip_id);
    auto* left_recv_device = DevicePool::instance().get_active_device(left_recv_phys_chip_id);
    auto* right_recv_device = DevicePool::instance().get_active_device(right_recv_phys_chip_id);

    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = left_recv_device->worker_core_from_logical_core(receiver_logical_core);

    auto receiver_noc_encoding =
        tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    const auto edm_config = get_tt_fabric_config();
    uint32_t is_2d_fabric = edm_config.topology == Topology::Mesh;

    // test parameters
    uint32_t packet_header_address = 0x25000;
    uint32_t source_l1_buffer_address = 0x30000;
    uint32_t packet_payload_size_bytes = edm_config.topology == Topology::Mesh ? 2048 : 4096;
    uint32_t num_packets = 100;
    uint32_t test_results_address = 0x100000;
    uint32_t test_results_size_bytes = 128;
    uint32_t target_address = 0x30000;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    // common compile time args for sender and receiver
    std::vector<uint32_t> compile_time_args = {
        test_results_address,
        test_results_size_bytes,
        target_address,
        1 /* mcast_mode */,
        edm_config.topology == Topology::Mesh};

    std::map<string, string> defines = {};
    if (is_2d_fabric) {
        defines["FABRIC_2D"] = "";
    }

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_tx.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args,
            .defines = defines});

    auto mesh_shape = control_plane->get_physical_mesh_shape(mesh_id.value());
    tt::log_info(tt::LogTest, "mesh dimensions {:x}", mesh_shape.dims());
    tt::log_info(tt::LogTest, "mesh dimension 0 {:x}", mesh_shape[0]);
    tt::log_info(tt::LogTest, "mesh dimension 1 {:x}", mesh_shape[1]);

    std::vector<uint32_t> sender_runtime_args = {
        packet_header_address,
        source_l1_buffer_address,
        packet_payload_size_bytes,
        num_packets,
        receiver_noc_encoding,
        time_seed,
        mesh_shape[1],
        src_chip_id,
        left_chip_id,
        1, /* mcast_fwd_hops */
    };

    // append the EDM connection rt args for fwd connection
    append_fabric_connection_rt_args(
        src_phys_chip_id, right_recv_phys_chip_id, 0, sender_program, {sender_logical_core}, sender_runtime_args);
    sender_runtime_args.push_back(right_chip_id);
    sender_runtime_args.push_back(1); /* mcast_bwd_hops */
    append_fabric_connection_rt_args(
        src_phys_chip_id, left_recv_phys_chip_id, 0, sender_program, {sender_logical_core}, sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    std::vector<uint32_t> receiver_runtime_args = {packet_payload_size_bytes, num_packets, time_seed};

    // Create the receiver program for validation
    auto receiver_program = tt_metal::CreateProgram();
    auto receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp",
        {receiver_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

    // Launch sender and receiver programs and wait for them to finish
    fixture->RunProgramNonblocking(left_recv_device, receiver_program);
    fixture->RunProgramNonblocking(right_recv_device, receiver_program);
    fixture->RunProgramNonblocking(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(right_recv_device, receiver_program);
    fixture->WaitForSingleProgramDone(left_recv_device, receiver_program);

    // Validate the status and packets processed by sender and receiver
    std::vector<uint32_t> sender_status;
    std::vector<uint32_t> left_recv_status;
    std::vector<uint32_t> right_recv_status;

    tt_metal::detail::ReadFromDeviceL1(
        sender_device,
        sender_logical_core,
        test_results_address,
        test_results_size_bytes,
        sender_status,
        CoreType::WORKER);

    tt_metal::detail::ReadFromDeviceL1(
        left_recv_device,
        receiver_logical_core,
        test_results_address,
        test_results_size_bytes,
        left_recv_status,
        CoreType::WORKER);

    tt_metal::detail::ReadFromDeviceL1(
        right_recv_device,
        receiver_logical_core,
        test_results_address,
        test_results_size_bytes,
        right_recv_status,
        CoreType::WORKER);

    EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
    EXPECT_EQ(left_recv_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
    EXPECT_EQ(right_recv_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);

    uint64_t sender_bytes =
        ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];
    uint64_t left_recv_bytes =
        ((uint64_t)left_recv_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | left_recv_status[TT_FABRIC_WORD_CNT_INDEX];
    uint64_t right_recv_bytes =
        ((uint64_t)right_recv_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | right_recv_status[TT_FABRIC_WORD_CNT_INDEX];

    EXPECT_EQ(sender_bytes, left_recv_bytes);
    EXPECT_EQ(left_recv_bytes, right_recv_bytes);
}

TEST_F(Fabric1DFixture, TestUnicastRaw) { RunTestUnicastRaw(this, 1); }
TEST_F(Fabric1DFixture, TestUnicastConnAPI) { RunTestUnicastConnAPI(this, 1); }
TEST_F(Fabric1DFixture, TestMCastConnAPI) { RunTestMCastConnAPI(this); }

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
