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
#include <random>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include "fabric_fixture.hpp"
#include "utils.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/fabric.hpp>
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "umd/device/tt_core_coordinates.h"

namespace tt::tt_fabric {
namespace fabric_router_tests {
std::random_device rd;  // Non-deterministic seed source
std::mt19937 global_rng(rd());

struct WorkerMemMap {
    uint32_t packet_header_address;
    uint32_t source_l1_buffer_address;
    uint32_t packet_payload_size_bytes;
    uint32_t test_results_address;
    uint32_t target_address;
    uint32_t test_results_size_bytes;
};

// Utility function reused across tests to get address params
WorkerMemMap generate_worker_mem_map(tt_metal::IDevice* device, Topology topology) {
    constexpr uint32_t PACKET_HEADER_RESERVED_BYTES = 45056;
    constexpr uint32_t DATA_SPACE_RESERVED_BYTES = 851968;
    constexpr uint32_t TEST_RESULTS_SIZE_BYTES = 128;

    uint32_t base_addr = device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
    uint32_t packet_header_address = base_addr;
    uint32_t source_l1_buffer_address = base_addr + PACKET_HEADER_RESERVED_BYTES;
    uint32_t test_results_address = source_l1_buffer_address + DATA_SPACE_RESERVED_BYTES;
    uint32_t target_address = source_l1_buffer_address;

    uint32_t packet_payload_size_bytes = (topology == Topology::Mesh) ? 2048 : 4096;

    return {
        packet_header_address,
        source_l1_buffer_address,
        packet_payload_size_bytes,
        test_results_address,
        target_address,
        TEST_RESULTS_SIZE_BYTES};
}

std::vector<uint32_t> get_random_numbers_from_range(uint32_t start, uint32_t end, uint32_t count) {
    std::vector<uint32_t> range(end - start + 1);

    // generate the range
    std::iota(range.begin(), range.end(), start);

    // shuffle the range
    std::shuffle(range.begin(), range.end(), global_rng);

    return std::vector<uint32_t>(range.begin(), range.begin() + count);
}

std::shared_ptr<tt_metal::Program> create_receiver_program(
    const std::vector<uint32_t>& compile_time_args,
    const std::vector<uint32_t>& runtime_args,
    const CoreCoord& logical_core) {
    auto recv_program = std::make_shared<tt_metal::Program>();
    auto recv_kernel = tt_metal::CreateKernel(
        *recv_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp",
        {logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});
    tt_metal::SetRuntimeArgs(*recv_program, recv_kernel, logical_core, runtime_args);
    return recv_program;
}

void RunTestLineMcast(
    BaseFabricFixture* fixture, RoutingDirection unicast_dir, const std::vector<McastRoutingInfo>& mcast_routing_info) {
    auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    auto user_meshes = control_plane.get_user_physical_mesh_ids();
    bool system_accomodates_mcast = false;
    for (const auto& mesh : user_meshes) {
        auto mesh_shape = control_plane.get_physical_mesh_shape(mesh);
        // Need at least 8 chips for all mcast tests
        if (mesh_shape.mesh_size() >= 8) {
            system_accomodates_mcast = true;
            break;
        }
    }
    if (!system_accomodates_mcast) {
        GTEST_SKIP() << "No mesh found for line mcast test";
    }
    // Setup mcast path
    chip_id_t mcast_start_phys_id;                              // Physical ID for chip starting mcast
    FabricNodeId mcast_start_id(MeshId{0}, 0);                          // Mesh ID for chip starting mcast
    std::unordered_map<RoutingDirection, uint32_t> mcast_hops;  // Specify mcast path from mcast src chip
    std::unordered_map<RoutingDirection, std::vector<FabricNodeId>>
        mcast_group;  // Mesh IDs for chips involved in mcast
    std::unordered_map<RoutingDirection, std::vector<chip_id_t>>
        mcast_group_phys_ids_per_dir;  // Physical IDs for chips involved in mcast

    for (const auto& routing_info : mcast_routing_info) {
        mcast_hops[routing_info.mcast_dir] = routing_info.num_mcast_hops;
    }

    bool mcast_group_found = find_device_with_neighbor_in_multi_direction(
        fixture,
        mcast_start_id,
        mcast_group,
        mcast_start_phys_id,
        mcast_group_phys_ids_per_dir,
        mcast_hops,
        unicast_dir);

    if (!mcast_group_found) {
        GTEST_SKIP() << "Mcast group not found for line mcast test";
    }

    // Compute coordinates of the remote chip that sends an mcast request to the mcast sender
    FabricNodeId sender_id =
        FabricNodeId(mcast_start_id.mesh_id, control_plane.get_intra_chip_neighbors(mcast_start_id, unicast_dir)[0]);
    auto sender_phys_id = control_plane.get_physical_chip_id_from_fabric_node_id(sender_id);
    // Compute physical IDs for mcast group chips
    std::vector<chip_id_t> mcast_group_phys_ids = {};
    for (const auto& routing_info : mcast_routing_info) {
        for (auto phys_id : mcast_group_phys_ids_per_dir[routing_info.mcast_dir]) {
            mcast_group_phys_ids.push_back(phys_id);
        }
    }

    CoreCoord sender_logical_core = {0, 0};    // This core on the sender (remote chip) will make the mcast request
    CoreCoord receiver_logical_core = {1, 0};  // Data will be forwarded to this core on al chips in the mcast group

    const auto& fabric_context = control_plane.get_fabric_context();
    const auto topology = fabric_context.get_fabric_topology();
    const auto& edm_config = fabric_context.get_fabric_router_config();
    uint32_t is_2d_fabric = edm_config.topology == Topology::Mesh;

    auto routers = control_plane.get_forwarding_eth_chans_to_chip(sender_id, mcast_start_id);
    if (routers.size() == 0) {
        log_info(
            tt::LogTest,
            "No fabric routers between Src MeshId {} ChipId {} - Dst MeshId {} ChipId {}",
            sender_id.mesh_id,
            sender_id.chip_id,
            mcast_start_id.mesh_id,
            mcast_start_id.chip_id);

        GTEST_SKIP() << "Skipping Test";
    }

    auto* sender_device = DevicePool::instance().get_active_device(sender_phys_id);
    auto* mcast_start_device = DevicePool::instance().get_active_device(mcast_start_phys_id);
    std::vector<tt_metal::IDevice*> mcast_group_devices = {};
    for (auto id : mcast_group_phys_ids) {
        mcast_group_devices.push_back(DevicePool::instance().get_active_device(id));
    }

    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = mcast_start_device->worker_core_from_logical_core(receiver_logical_core);

    auto receiver_noc_encoding =
        tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    auto mesh_shape = control_plane.get_physical_mesh_shape(sender_id.mesh_id);

    auto worker_mem_map = generate_worker_mem_map(sender_device, edm_config.topology);
    uint32_t num_packets = 100;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    // common compile time args for sender and receiver
    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address, worker_mem_map.test_results_size_bytes, worker_mem_map.target_address};

    std::map<string, string> defines = {};
    if (is_2d_fabric) {
        defines["FABRIC_2D"] = "";
    }

    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_line_mcast_tx.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args,
            .defines = defines});

    std::vector<uint32_t> sender_runtime_args = {
        worker_mem_map.packet_header_address,
        worker_mem_map.source_l1_buffer_address,
        worker_mem_map.packet_payload_size_bytes,
        num_packets,
        receiver_noc_encoding,
        time_seed,
        mcast_start_id.chip_id,
        *mcast_start_id.mesh_id};

    std::vector<uint32_t> mcast_header_rtas(4, 0);
    for (const auto& routing_info : mcast_routing_info) {
        mcast_header_rtas[static_cast<uint32_t>(
            control_plane.routing_direction_to_eth_direction(routing_info.mcast_dir))] = routing_info.num_mcast_hops;
    }
    sender_runtime_args.insert(sender_runtime_args.end(), mcast_header_rtas.begin(), mcast_header_rtas.end());
    // append the EDM connection rt args
    append_fabric_connection_rt_args(
        sender_phys_id, mcast_start_phys_id, 0, sender_program, {sender_logical_core}, sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Create the receiver programs for validation on all devices involved in the Mcast
    std::vector<uint32_t> receiver_runtime_args = {worker_mem_map.packet_payload_size_bytes, num_packets, time_seed};
    std::unordered_map<tt_metal::IDevice*, std::shared_ptr<tt_metal::Program>> recv_programs;
    recv_programs[mcast_start_device] =
        create_receiver_program(compile_time_args, receiver_runtime_args, receiver_logical_core);
    for (const auto& dev : mcast_group_devices) {
        recv_programs[dev] = create_receiver_program(compile_time_args, receiver_runtime_args, receiver_logical_core);
    }

    // Launch sender and receiver programs and wait for them to finish
    for (auto& [dev, recv_program] : recv_programs) {
        log_info(tt::LogTest, "Run receiver on: {}", dev->id());
        fixture->RunProgramNonblocking(dev, *recv_program);
    }
    log_info(tt::LogTest, "Run Sender on: {}", sender_device->id());
    fixture->RunProgramNonblocking(sender_device, sender_program);

    for (auto& [dev, recv_program] : recv_programs) {
        fixture->WaitForSingleProgramDone(dev, *recv_program);
    }
    fixture->WaitForSingleProgramDone(sender_device, sender_program);

    std::vector<uint32_t> sender_status;
    tt_metal::detail::ReadFromDeviceL1(
        sender_device,
        sender_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        sender_status,
        CoreType::WORKER);

    EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
    uint64_t sender_bytes =
        ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];

    for (auto& [dev, _] : recv_programs) {
        std::vector<uint32_t> receiver_status;
        tt_metal::detail::ReadFromDeviceL1(
            dev,
            receiver_logical_core,
            worker_mem_map.test_results_address,
            worker_mem_map.test_results_size_bytes,
            receiver_status,
            CoreType::WORKER);

        EXPECT_EQ(receiver_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
        uint64_t receiver_bytes =
            ((uint64_t)receiver_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | receiver_status[TT_FABRIC_WORD_CNT_INDEX];

        EXPECT_EQ(sender_bytes, receiver_bytes);
    }
}

void RunTestUnicastRaw(
    BaseFabricFixture* fixture, uint32_t num_hops, RoutingDirection direction, bool enable_fabric_tracing) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};

    auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();

    FabricNodeId src_fabric_node_id(MeshId{0}, 0);
    FabricNodeId dst_fabric_node_id(MeshId{0}, 0);
    chip_id_t not_used_1;
    chip_id_t not_used_2;
    // Find a device num_hops away in specified direction.
    std::unordered_map<RoutingDirection, uint32_t> fabric_hops;
    std::unordered_map<RoutingDirection, std::vector<FabricNodeId>> end_fabric_node_ids_by_dir;
    chip_id_t src_physical_device_id;
    chip_id_t dst_physical_device_id;
    std::unordered_map<RoutingDirection, std::vector<chip_id_t>> physical_end_device_ids_by_dir;
    fabric_hops[direction] = num_hops;

    tt::tt_metal::distributed::MeshShape mesh_shape;
    std::vector<chan_id_t> eth_chans;
    chan_id_t edm_port;

    const auto& fabric_context = control_plane.get_fabric_context();
    const auto topology = fabric_context.get_fabric_topology();
    const auto& edm_config = fabric_context.get_fabric_router_config();
    uint32_t is_2d_fabric = topology == Topology::Mesh;

    if (!is_2d_fabric) {
        // Find a device with enough neighbours in the specified directions
        if (!find_device_with_neighbor_in_multi_direction(
                fixture,
                src_fabric_node_id,
                end_fabric_node_ids_by_dir,
                src_physical_device_id,
                physical_end_device_ids_by_dir,
                fabric_hops)) {
            GTEST_SKIP() << "No path found between sender and receivers";
        }
        mesh_shape = control_plane.get_physical_mesh_shape(src_fabric_node_id.mesh_id);
        dst_physical_device_id = physical_end_device_ids_by_dir[direction][num_hops - 1];
        dst_fabric_node_id = end_fabric_node_ids_by_dir[direction][num_hops - 1];

        // get a port to connect to
        eth_chans = control_plane.get_active_fabric_eth_channels_in_direction(src_fabric_node_id, direction);
        if (eth_chans.size() == 0) {
            GTEST_SKIP() << "No active eth chans to connect to";
        }
    } else {
        auto devices = fixture->get_devices();
        auto num_devices = devices.size();
        // create a list of available deive ids in a random order
        // In 2D routing the source and desitnation devices can be anywhere on the mesh.
        auto random_dev_list = get_random_numbers_from_range(0, devices.size() - 1, devices.size());

        // pick the first two in the list to be src and dst devices for the test.
        src_physical_device_id = devices[random_dev_list[0]]->id();
        dst_physical_device_id = devices[random_dev_list[1]]->id();
        src_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(src_physical_device_id);
        dst_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(dst_physical_device_id);
        mesh_shape = control_plane.get_physical_mesh_shape(src_fabric_node_id.mesh_id);

        eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_fabric_node_id, dst_fabric_node_id);
        if (eth_chans.size() == 0) {
            log_info(
                tt::LogTest,
                "No fabric routers between Src MeshId {} ChipId {} - Dst MeshId {} ChipId {}",
                src_fabric_node_id.mesh_id,
                src_fabric_node_id.chip_id,
                *dst_fabric_node_id.mesh_id,
                dst_fabric_node_id.chip_id);

            GTEST_SKIP() << "Skipping Test";
        }
    }

    // Pick any port, for now pick the 1st one in the set
    edm_port = *eth_chans.begin();

    log_info(tt::LogTest, "mesh dimensions {:x}", mesh_shape.dims());
    log_info(tt::LogTest, "mesh size {:x}", mesh_shape.mesh_size());
    log_info(tt::LogTest, "mesh dimension 0 {:x}", mesh_shape[0]);
    log_info(tt::LogTest, "mesh dimension 1 {:x}", mesh_shape[1]);
    log_info(tt::LogTest, "Src MeshId {} ChipId {}", src_fabric_node_id.mesh_id, src_fabric_node_id.chip_id);
    log_info(tt::LogTest, "Dst MeshId {} ChipId {}", dst_fabric_node_id.mesh_id, dst_fabric_node_id.chip_id);

    auto edm_direction = control_plane.get_eth_chan_direction(src_fabric_node_id, edm_port);
    CoreCoord edm_eth_core = tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_eth_core_from_channel(
        src_physical_device_id, edm_port);

    log_info(tt::LogTest, "Using edm port {} in direction {}", edm_port, edm_direction);

    auto* sender_device = DevicePool::instance().get_active_device(src_physical_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(dst_physical_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    auto receiver_noc_encoding =
        tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // test parameters
    auto worker_mem_map = generate_worker_mem_map(sender_device, topology);
    uint32_t num_packets = 10;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    const auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();

    // common compile time args for sender and receiver
    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        worker_mem_map.target_address,
        0 /* mcast_mode */,
        topology == Topology::Mesh,
        fabric_config == tt_metal::FabricConfig::FABRIC_2D_DYNAMIC};

    std::map<string, string> defines = {};
    if (is_2d_fabric) {
        defines["FABRIC_2D"] = "";
    }

    if (enable_fabric_tracing) {
        defines["TEST_ENABLE_FABRIC_TRACING"] = "1";
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

    std::vector<uint32_t> sender_runtime_args = {
        worker_mem_map.packet_header_address,
        worker_mem_map.source_l1_buffer_address,
        worker_mem_map.packet_payload_size_bytes,
        num_packets,
        receiver_noc_encoding,
        time_seed,
        mesh_shape[1],
        src_fabric_node_id.chip_id,
        dst_fabric_node_id.chip_id,
        *dst_fabric_node_id.mesh_id,
        num_hops};

    // append the EDM connection rt args
    const auto sender_channel = topology == Topology::Mesh ? edm_direction : 0;
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

    std::vector<uint32_t> receiver_runtime_args = {worker_mem_map.packet_payload_size_bytes, num_packets, time_seed};

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

    if (enable_fabric_tracing) {
        tt_metal::detail::DumpDeviceProfileResults(sender_device);
    }

    // Validate the status and packets processed by sender and receiver
    std::vector<uint32_t> sender_status;
    std::vector<uint32_t> receiver_status;

    tt_metal::detail::ReadFromDeviceL1(
        sender_device,
        sender_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        sender_status,
        CoreType::WORKER);

    tt_metal::detail::ReadFromDeviceL1(
        receiver_device,
        receiver_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
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

void run_unicast_test_bw_chips(
    BaseFabricFixture* fixture, chip_id_t src_physical_device_id, chip_id_t dst_physical_device_id, uint32_t num_hops) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};

    const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    auto src_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(src_physical_device_id);
    auto dst_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(dst_physical_device_id);

    auto* sender_device = DevicePool::instance().get_active_device(src_physical_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(dst_physical_device_id);
    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    auto receiver_noc_encoding =
        tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    const auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
    const auto topology = control_plane.get_fabric_context().get_fabric_topology();
    uint32_t is_2d_fabric = topology == Topology::Mesh;

    // test parameters
    auto worker_mem_map = generate_worker_mem_map(sender_device, topology);
    uint32_t num_packets = 10;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    // common compile time args for sender and receiver
    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        worker_mem_map.target_address,
        0 /* mcast_mode */,
        topology == Topology::Mesh,
        fabric_config == tt_metal::FabricConfig::FABRIC_2D_DYNAMIC};

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

    auto mesh_shape = control_plane.get_physical_mesh_shape(src_fabric_node_id.mesh_id);
    log_info(tt::LogTest, "mesh dimensions {:x}", mesh_shape.dims());
    log_info(tt::LogTest, "mesh dimension 0 {:x}", mesh_shape[0]);
    log_info(tt::LogTest, "mesh dimension 1 {:x}", mesh_shape[1]);

    std::vector<uint32_t> sender_runtime_args = {
        worker_mem_map.packet_header_address,
        worker_mem_map.source_l1_buffer_address,
        worker_mem_map.packet_payload_size_bytes,
        num_packets,
        receiver_noc_encoding,
        time_seed,
        mesh_shape[1],
        src_fabric_node_id.chip_id,
        dst_fabric_node_id.chip_id,
        *dst_fabric_node_id.mesh_id,
        num_hops};

    // append the EDM connection rt args
    const auto& available_links = get_forwarding_link_indices(src_physical_device_id, dst_physical_device_id);
    EXPECT_EQ(available_links.size() > 0, true);

    uint32_t link_idx = available_links[0];
    append_fabric_connection_rt_args(
        src_physical_device_id,
        dst_physical_device_id,
        link_idx,
        sender_program,
        {sender_logical_core},
        sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    std::vector<uint32_t> receiver_runtime_args = {worker_mem_map.packet_payload_size_bytes, num_packets, time_seed};

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
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        sender_status,
        CoreType::WORKER);

    tt_metal::detail::ReadFromDeviceL1(
        receiver_device,
        receiver_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
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
    const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();

    FabricNodeId src_fabric_node_id(MeshId{0}, 0);
    FabricNodeId dst_fabric_node_id(MeshId{0}, 0);
    chip_id_t not_used_1;
    chip_id_t not_used_2;
    // Find a device with a neighbour in the East direction
    bool connection_found = find_device_with_neighbor_in_direction(
        fixture, src_fabric_node_id, dst_fabric_node_id, not_used_1, not_used_2, direction);
    if (!connection_found) {
        GTEST_SKIP() << "No path found between sender and receivers";
    }

    log_info(tt::LogTest, "Src MeshId {} ChipId {}", src_fabric_node_id.mesh_id, src_fabric_node_id.chip_id);
    log_info(tt::LogTest, "Dst MeshId {} ChipId {}", dst_fabric_node_id.mesh_id, dst_fabric_node_id.chip_id);
    log_info(tt::LogTest, "Dst Device is {} hops in direction: {}", num_hops, direction);

    chip_id_t src_physical_device_id = control_plane.get_physical_chip_id_from_fabric_node_id(src_fabric_node_id);
    chip_id_t dst_physical_device_id = control_plane.get_physical_chip_id_from_fabric_node_id(dst_fabric_node_id);

    run_unicast_test_bw_chips(fixture, src_physical_device_id, dst_physical_device_id, num_hops);
}

void RunTestUnicastConnAPIRandom(BaseFabricFixture* fixture) {
    const auto topology = tt::tt_metal::MetalContext::instance()
                              .get_control_plane()
                              .get_fabric_context()
                              .get_fabric_topology();
    uint32_t is_2d_fabric = topology == Topology::Mesh;
    if (!is_2d_fabric) {
        GTEST_SKIP() << "This test is only supported for 2D fabric currently";
    }

    auto devices = fixture->get_devices();
    // create a list of available deive ids in a random order
    // In 2D routing the source and desitnation devices can be anywhere on the mesh.
    auto random_dev_list = get_random_numbers_from_range(0, devices.size() - 1, 2);

    const auto src_physical_device_id = devices[random_dev_list[0]]->id();
    const auto dst_physical_device_id = devices[random_dev_list[1]]->id();

    log_info(tt::LogTest, "Src Phys ChipId {}", src_physical_device_id);
    log_info(tt::LogTest, "Dst Phys ChipId {}", dst_physical_device_id);

    run_unicast_test_bw_chips(
        fixture, src_physical_device_id, dst_physical_device_id, 0 /* num_hops, not needed for 2d */);
}

void RunTestUnicastTGGateways(BaseFabricFixture* fixture) {
    // TODO: remove this restriction once tunneling is disabled
    if (!fixture->slow_dispatch_) {
        log_info(tt::LogTest, "This test can only be run with TT_METAL_SLOW_DISPATCH_MODE currently");
        GTEST_SKIP();
    }

    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::ClusterType::TG) {
        log_info(tt::LogTest, "This test is only for TG");
        GTEST_SKIP();
    }

    // run tests b/w all pairs of TG gateways <> remote chip connections
    // this only tests connections with the immediate remote chip in the tunnel since other connections
    // are 'normal' and covered in other tests
    const std::vector<chip_id_t> mmio_chip_ids = {0, 1, 2, 3};
    for (const auto& mmio_chip_id : mmio_chip_ids) {
        const auto& tunnels_from_mmio =
            tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_chip_id);
        for (uint32_t t = 0; t < tunnels_from_mmio.size(); t++) {
            // idx 0 in the tunnel is the mmio chip itself
            const auto remote_chip_id = tunnels_from_mmio[t][1];
            log_info(tt::LogTest, "Running tests for chips: {} and {}", mmio_chip_id, remote_chip_id);
            run_unicast_test_bw_chips(fixture, mmio_chip_id, remote_chip_id, 1);
            run_unicast_test_bw_chips(fixture, remote_chip_id, mmio_chip_id, 1);
        }
    }
}

void RunTestMCastConnAPI(
    BaseFabricFixture* fixture,
    RoutingDirection fwd_dir,
    uint32_t fwd_hops,
    RoutingDirection bwd_dir,
    uint32_t bwd_hops) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};
    std::vector<tt_metal::Program> receiver_programs;

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // use control plane to find a mesh with 3 devices
    auto user_meshes = control_plane.get_user_physical_mesh_ids();
    std::optional<MeshId> mesh_id;
    for (const auto& mesh : user_meshes) {
        auto mesh_shape = control_plane.get_physical_mesh_shape(mesh);
        if (mesh_shape.mesh_size() >= 3) {
            mesh_id = mesh;
            break;
        }
    }
    if (!mesh_id.has_value()) {
        GTEST_SKIP() << "No mesh found for 3 chip mcast test";
    }

    // Find a device num_hops away in specified direction.
    FabricNodeId src_fabric_node_id(MeshId{0}, 0);
    std::unordered_map<RoutingDirection, uint32_t> fabric_hops;
    std::unordered_map<RoutingDirection, std::vector<FabricNodeId>> end_fabric_node_ids_by_dir;
    chip_id_t src_phys_chip_id;
    std::unordered_map<RoutingDirection, std::vector<chip_id_t>> physical_end_device_ids_by_dir;
    fabric_hops[fwd_dir] = fwd_hops;
    fabric_hops[bwd_dir] = bwd_hops;

    tt::tt_metal::distributed::MeshShape mesh_shape;
    const auto topology = control_plane.get_fabric_context().get_fabric_topology();
    uint32_t is_2d_fabric = topology == Topology::Mesh;

    // Get the mcast sender device and mcast receiver devices that satisfy the input number of hops in forward and
    // backward directions.
    if (!find_device_with_neighbor_in_multi_direction(
            fixture,
            src_fabric_node_id,
            end_fabric_node_ids_by_dir,
            src_phys_chip_id,
            physical_end_device_ids_by_dir,
            fabric_hops)) {
        log_info(
            tt::LogTest,
            "No Mcast destinations found for {} hops in {} and {} hops in {}",
            fwd_hops,
            fwd_dir,
            bwd_hops,
            bwd_dir);
        GTEST_SKIP() << "Skipping Test";
    }

    mesh_shape = control_plane.get_physical_mesh_shape(src_fabric_node_id.mesh_id);
    auto left_recv_phys_chip_id = physical_end_device_ids_by_dir[fwd_dir][fwd_hops - 1];
    auto left_first_hop_phys_chip_id = physical_end_device_ids_by_dir[fwd_dir][0];
    auto right_recv_phys_chip_id = physical_end_device_ids_by_dir[bwd_dir][bwd_hops - 1];
    auto right_first_hop_phys_chip_id = physical_end_device_ids_by_dir[bwd_dir][0];

    auto* sender_device = DevicePool::instance().get_active_device(src_phys_chip_id);
    auto* left_recv_device = DevicePool::instance().get_active_device(left_recv_phys_chip_id);
    auto* right_recv_device = DevicePool::instance().get_active_device(right_recv_phys_chip_id);

    auto left_fabric_node_id = end_fabric_node_ids_by_dir[fwd_dir][fwd_hops - 1];
    auto right_fabric_node_id = end_fabric_node_ids_by_dir[bwd_dir][bwd_hops - 1];

    CoreCoord sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
    CoreCoord receiver_virtual_core = left_recv_device->worker_core_from_logical_core(receiver_logical_core);

    auto receiver_noc_encoding =
        tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // test parameters
    auto worker_mem_map = generate_worker_mem_map(sender_device, topology);
    uint32_t num_packets = 100;
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    const auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();

    // common compile time args for sender and receiver
    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        worker_mem_map.target_address,
        1 /* mcast_mode */,
        topology == Topology::Mesh,
        fabric_config == tt_metal::FabricConfig::FABRIC_2D_DYNAMIC};

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

    log_info(tt::LogTest, "mesh dimensions {:x}", mesh_shape.dims());
    log_info(tt::LogTest, "mesh dimension 0 {:x}", mesh_shape[0]);
    log_info(tt::LogTest, "mesh dimension 1 {:x}", mesh_shape[1]);
    log_info(tt::LogTest, "Mcast Src MeshId {} ChipId {}", src_fabric_node_id.mesh_id, src_fabric_node_id.chip_id);
    log_info(
        tt::LogTest, "Mcast Fwd Dst MeshId {} ChipId {}", left_fabric_node_id.mesh_id, left_fabric_node_id.chip_id);
    log_info(tt::LogTest, "Mcast Fwd Dst Device is {} hops in direction: {}", fwd_hops, fwd_dir);
    log_info(
        tt::LogTest, "Mcast Bwd Dst MeshId {} ChipId {}", right_fabric_node_id.mesh_id, right_fabric_node_id.chip_id);
    log_info(tt::LogTest, "Mcast Bwd Dst Device is {} hops in direction: {}", bwd_hops, bwd_dir);

    std::vector<uint32_t> sender_runtime_args = {
        worker_mem_map.packet_header_address,
        worker_mem_map.source_l1_buffer_address,
        worker_mem_map.packet_payload_size_bytes,
        num_packets,
        receiver_noc_encoding,
        time_seed,
        mesh_shape[1],
        src_fabric_node_id.chip_id,
        left_fabric_node_id.chip_id,
        *mesh_id.value(),
        fwd_hops, /* mcast_fwd_hops */
    };

    // append the EDM connection rt args for fwd connection
    chip_id_t dst_chip_id;
    uint32_t link_idx;

    if (is_2d_fabric) {
        dst_chip_id = left_recv_phys_chip_id;
    } else {
        dst_chip_id = left_first_hop_phys_chip_id;
    }
    link_idx = get_forwarding_link_indices(src_phys_chip_id, dst_chip_id)[0];
    append_fabric_connection_rt_args(
        src_phys_chip_id, dst_chip_id, link_idx, sender_program, {sender_logical_core}, sender_runtime_args);
    sender_runtime_args.push_back(right_fabric_node_id.chip_id);
    sender_runtime_args.push_back(bwd_hops); /* mcast_bwd_hops */

    if (is_2d_fabric) {
        dst_chip_id = right_recv_phys_chip_id;
    } else {
        dst_chip_id = right_first_hop_phys_chip_id;
    }
    link_idx = get_forwarding_link_indices(src_phys_chip_id, dst_chip_id)[0];
    append_fabric_connection_rt_args(
        src_phys_chip_id, dst_chip_id, link_idx, sender_program, {sender_logical_core}, sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    std::vector<uint32_t> receiver_runtime_args = {worker_mem_map.packet_payload_size_bytes, num_packets, time_seed};

    // Create and launch the receiver program for validation on all mcast receiver devices
    for (auto& [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (auto physical_end_device_id : physical_end_device_ids) {
            auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_id);
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
            fixture->RunProgramNonblocking(receiver_device, receiver_program);
            receiver_programs.push_back(std::move(receiver_program));
            log_info(tt::LogTest, "{} Rx Launched on physical device {}", routing_direction, physical_end_device_id);
        }
    }

    // Launch sender program and wait for sender to finish
    fixture->RunProgramNonblocking(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(sender_device, sender_program);

    // Wait for receivers to finish
    for (auto [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (uint32_t i = 0; i < physical_end_device_ids.size(); i++) {
            auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_ids[i]);
            fixture->WaitForSingleProgramDone(receiver_device, receiver_programs[i]);
        }
    }
    log_info(tt::LogTest, "All Receivers Finished");

    // Validate the status and packets processed by sender and receiver
    std::vector<uint32_t> sender_status;
    std::vector<uint32_t> left_recv_status;
    std::vector<uint32_t> right_recv_status;

    tt_metal::detail::ReadFromDeviceL1(
        sender_device,
        sender_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        sender_status,
        CoreType::WORKER);

    EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);

    uint64_t sender_bytes =
        ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];

    for (auto [routing_direction, physical_end_device_ids] : physical_end_device_ids_by_dir) {
        for (uint32_t i = 0; i < physical_end_device_ids.size(); i++) {
            log_info(
                tt::LogTest,
                "Checking Status of {} Rx on physical device {}",
                routing_direction,
                physical_end_device_ids[i]);

            auto* receiver_device = DevicePool::instance().get_active_device(physical_end_device_ids[i]);
            std::vector<uint32_t> recv_status;

            tt_metal::detail::ReadFromDeviceL1(
                receiver_device,
                receiver_logical_core,
                worker_mem_map.test_results_address,
                worker_mem_map.test_results_size_bytes,
                recv_status,
                CoreType::WORKER);
            EXPECT_EQ(recv_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
            uint64_t recv_bytes =
                ((uint64_t)recv_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | recv_status[TT_FABRIC_WORD_CNT_INDEX];

            EXPECT_EQ(sender_bytes, recv_bytes);
        }
    }
}

TEST_F(Fabric1DFixture, TestUnicastRaw) { RunTestUnicastRaw(this, 1, RoutingDirection::E, false); }
TEST_F(Fabric1DFixture, TestUnicastConnAPI) { RunTestUnicastConnAPI(this, 1); }
TEST_F(Fabric1DFixture, TestUnicastTGGateways) { RunTestUnicastTGGateways(this); }
TEST_F(Fabric1DFixture, TestMCastConnAPI) { RunTestMCastConnAPI(this); }

TEST_F(Fabric1DFixture, TestUnicastRawWithTracing) { RunTestUnicastRaw(this, 1, RoutingDirection::E, true); }

TEST_F(Fabric1DFixture, DISABLED_TestEDMConnectionStressTestQuick) {
    // Each epoch is a separate program launch with increasing number of workers
    std::vector<size_t> stall_durations_cycles = {0,    100,  200,  300,   400,   700,   1000,  2000,  3000,  4000,
                                                  5000, 7000, 8000, 10000, 20000, 30000, 40000, 50000, 60000, 100000};

    std::vector<size_t> message_counts = {8, 100};
    std::vector<size_t> packet_sizes = {16, 4 * 1088};
    size_t num_epochs = 5;
    size_t num_times_to_connect = 20000;  // How many times each worker connects during its turn

    log_debug(tt::LogTest, "Starting EDM connection stress test");
    auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    log_debug(tt::LogTest, "Control plane found");

    std::pair<MeshId, chip_id_t> src_mesh_chip_id;
    std::pair<MeshId, chip_id_t> dst_mesh_chip_id;
    chip_id_t not_used_1;
    chip_id_t not_used_2;
    // use control plane to find a mesh with 3 devices
    auto user_meshes = control_plane.get_user_physical_mesh_ids();
    std::optional<MeshId> mesh_id;
    for (const auto& mesh : user_meshes) {
        auto mesh_shape = control_plane.get_physical_mesh_shape(mesh);
        if (mesh_shape.mesh_size() > 1) {
            mesh_id = mesh;
            break;
        }
    }
    if (!mesh_id.has_value()) {
        GTEST_SKIP() << "No mesh found for 2 chip connection stress test";
    }

    log_debug(tt::LogTest, "Mesh ID: {}", mesh_id.value());
    auto src_physical_device_id =
        control_plane.get_physical_chip_id_from_fabric_node_id(FabricNodeId(mesh_id.value(), 0));
    auto dst_physical_device_id =
        control_plane.get_physical_chip_id_from_fabric_node_id(FabricNodeId(mesh_id.value(), 1));

    auto* sender_device = DevicePool::instance().get_active_device(src_physical_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(dst_physical_device_id);

    // Set the destination address for fabric writes (constant for all workers)
    uint32_t fabric_write_dest_bank_addr = 0x50000;

    // For each epoch, run with increasing number of workers
    log_debug(tt::LogTest, "Starting EDM connection stress test");
    auto compute_with_storage_grid_size = sender_device->compute_with_storage_grid_size();
    size_t num_rows = compute_with_storage_grid_size.y;
    size_t num_cols = compute_with_storage_grid_size.x;
    for (size_t iter = 0; iter < 10; iter++) {
        log_debug(tt::LogTest, "iter {}", iter);
        for (size_t num_workers : {1, 3}) {
            log_debug(tt::LogTest, "num_workers {}", num_workers);
            for (size_t r : {0, 4, 5, 6}) {
                for (size_t c = 0; c < num_cols - (num_workers - 1); c++) {
                    log_debug(tt::LogTest, "r={}, c={}", r, c);

                    // Set up worker cores for token ring
                    auto worker_logical_cores = CoreRangeSet(CoreRange({{c, r}, {c + num_workers - 1, r}}));
                    auto worker_logical_cores_vec = corerange_to_cores(worker_logical_cores, std::nullopt, false);

                    // Map logical to virtual cores
                    std::vector<CoreCoord> worker_virtual_cores;
                    for (const auto& logical_core : worker_logical_cores_vec) {
                        worker_virtual_cores.push_back(sender_device->worker_core_from_logical_core(logical_core));
                    }

                    // Create program
                    auto program = tt_metal::CreateProgram();

                    // Create semaphores for token passing (one per worker)
                    auto connection_token_semaphore_id =
                        tt_metal::CreateSemaphore(program, CoreRangeSet(worker_logical_cores), 0);

                    // Create source packet buffer (one per worker)
                    static constexpr uint32_t source_l1_cb_index = tt::CB::c_in0;
                    static constexpr uint32_t packet_header_cb_index = tt::CB::c_in1;
                    static constexpr tt::DataFormat cb_df = tt::DataFormat::Bfp8;
                    auto max_payload_size = *std::max_element(packet_sizes.begin(), packet_sizes.end());
                    auto source_l1_cb_config =
                        tt_metal::CircularBufferConfig(max_payload_size * 2, {{source_l1_cb_index, cb_df}})
                            .set_page_size(source_l1_cb_index, max_payload_size);
                    CreateCircularBuffer(program, worker_logical_cores, source_l1_cb_config);

                    // Create packet header buffer (one per worker)
                    auto packet_header_cb_config =
                        tt_metal::CircularBufferConfig(1024, {{packet_header_cb_index, cb_df}})
                            .set_page_size(packet_header_cb_index, 32);
                    CreateCircularBuffer(program, worker_logical_cores, packet_header_cb_config);

                    // Configure common compile time args for all workers
                    std::vector<uint32_t> compile_time_args = {
                        static_cast<uint32_t>(stall_durations_cycles.size()),
                        static_cast<uint32_t>(packet_sizes.size()),
                        static_cast<uint32_t>(message_counts.size()),
                    };

                    // Create a kernel for each worker
                    std::vector<std::vector<uint32_t>> runtime_args_per_worker(num_workers);

                    for (size_t i = 0; i < num_workers; i++) {
                        // Compute destination NOC coordinates for this worker
                        auto dest_virtual_core = worker_virtual_cores[i];

                        // Compute next worker index in the token ring
                        size_t next_worker_idx = (i + 1) % num_workers;
                        auto next_worker_virtual_core = worker_virtual_cores[next_worker_idx];

                        // Prepare runtime args for this worker
                        std::vector<uint32_t>& worker_args = runtime_args_per_worker[i];

                        // Basic configuration
                        worker_args.push_back(fabric_write_dest_bank_addr);  // Fabric write destination bank address
                        worker_args.push_back(dest_virtual_core.x);          // Fabric write destination NOC X
                        worker_args.push_back(dest_virtual_core.y);          // Fabric write destination NOC Y

                        // Token ring configuration
                        worker_args.push_back(i == 0 ? 1 : 0);              // Is starting worker (first worker starts)
                        worker_args.push_back(num_times_to_connect);        // How many times to connect during turn
                        worker_args.push_back(next_worker_virtual_core.x);  // Next worker NOC X
                        worker_args.push_back(next_worker_virtual_core.y);  // Next worker NOC Y
                        worker_args.push_back(connection_token_semaphore_id);  // Address of next worker's token

                        // Traffic pattern arrays (rotate starting index by worker ID for variation)
                        worker_args.push_back(stall_durations_cycles.size());  // Number of stall durations

                        // Rotate starting point for each worker to prevent lock-step behavior
                        size_t stall_offset = i % stall_durations_cycles.size();
                        for (size_t j = 0; j < stall_durations_cycles.size(); j++) {
                            size_t idx = (stall_offset + j) % stall_durations_cycles.size();
                            worker_args.push_back(stall_durations_cycles[idx]);
                        }

                        worker_args.push_back(packet_sizes.size());  // Number of packet sizes
                        size_t packet_size_offset = i % packet_sizes.size();
                        for (size_t j = 0; j < packet_sizes.size(); j++) {
                            size_t idx = (packet_size_offset + j) % packet_sizes.size();
                            worker_args.push_back(packet_sizes[idx]);
                        }

                        worker_args.push_back(message_counts.size());  // Number of message counts
                        size_t message_count_offset = i % message_counts.size();
                        for (size_t j = 0; j < message_counts.size(); j++) {
                            size_t idx = (message_count_offset + j) % message_counts.size();
                            worker_args.push_back(message_counts[idx]);
                        }

                        // Circular buffer indices for source data and packet headers
                        worker_args.push_back(source_l1_cb_index);      // Source L1 circular buffer index
                        worker_args.push_back(packet_header_cb_index);  // Packet header circular buffer index
                        worker_args.push_back(1);                       // Number of headers (size units in words)

                        worker_args.push_back(i % stall_durations_cycles.size());
                        worker_args.push_back(i % packet_sizes.size());
                        worker_args.push_back(i % message_counts.size());

                        append_fabric_connection_rt_args(
                            sender_device->id(),
                            receiver_device->id(),
                            0,
                            program,
                            {worker_logical_cores_vec[i]},
                            worker_args);

                        auto kernel = tt_metal::CreateKernel(
                            program,
                            "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/"
                            "edm_fabric_connection_test_kernel.cpp",
                            worker_logical_cores_vec[i],
                            tt_metal::DataMovementConfig{
                                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                                .noc = tt_metal::NOC::RISCV_0_default,
                                .compile_args = compile_time_args});
                        tt_metal::SetRuntimeArgs(
                            program, kernel, worker_logical_cores_vec[i], runtime_args_per_worker[i]);
                    }

                    // Launch program and wait for completion
                    auto start_time = std::chrono::high_resolution_clock::now();
                    log_debug(tt::LogTest, "Launching program");
                    this->RunProgramNonblocking(sender_device, program);
                    this->WaitForSingleProgramDone(sender_device, program);
                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto duration_ms =
                        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

                    log_debug(
                        tt::LogTest, "Iter {} with {} workers completed in {} ms", iter, num_workers, duration_ms);
                }
            }
        }
    }
}

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
