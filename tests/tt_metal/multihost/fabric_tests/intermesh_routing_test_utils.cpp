// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>

#include <iostream>
#include <random>
#include <stdint.h>

#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/fabric_host_interface.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/host_api.hpp>

#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/fabric_context.hpp"
#include "intermesh_routing_test_utils.hpp"

namespace tt::tt_fabric {
namespace fabric_router_tests {

namespace multihost_utils {
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
WorkerMemMap generate_worker_mem_map(tt_metal::IDevice* device) {
    constexpr uint32_t PACKET_HEADER_RESERVED_BYTES = 45056;
    constexpr uint32_t DATA_SPACE_RESERVED_BYTES = 851968;
    constexpr uint32_t TEST_RESULTS_SIZE_BYTES = 128;

    uint32_t base_addr = device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
    uint32_t packet_header_address = base_addr;
    uint32_t source_l1_buffer_address = base_addr + PACKET_HEADER_RESERVED_BYTES;
    uint32_t test_results_address = source_l1_buffer_address + DATA_SPACE_RESERVED_BYTES;
    uint32_t target_address = source_l1_buffer_address;

    uint32_t packet_payload_size_bytes = 2048;

    return {
        packet_header_address,
        source_l1_buffer_address,
        packet_payload_size_bytes,
        test_results_address,
        target_address,
        TEST_RESULTS_SIZE_BYTES};
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

void run_unicast_sender_step(BaseFabricFixture* fixture, tt::tt_metal::distributed::multihost::Rank recv_host_rank) {
    // The following code runs on the sender host
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();

    constexpr uint32_t num_packets = 100;

    const auto& fabric_context = control_plane.get_fabric_context();
    const auto topology = fabric_context.get_fabric_topology();
    TT_FATAL(topology == Topology::Mesh, "Intermesh Routing tests need Dynamic Routing enabled.");
    const auto& edm_config = fabric_context.get_fabric_router_config();

    auto devices = fixture->get_devices();
    auto num_devices = devices.size();
    const auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();

    // Synchronize seeds across hosts (sender and receiver must use the same seed for randomization)
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();
    distributed_context->send(
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&time_seed), sizeof(time_seed)),
        tt::tt_metal::distributed::multihost::Rank{recv_host_rank},  // send to receiver host
        tt::tt_metal::distributed::multihost::Tag{0}                 // exchange seed over tag 0
    );
    // Randomly select a tx device
    auto random_dev = std::uniform_int_distribution<uint32_t>(0, devices.size() - 1)(global_rng);
    auto src_physical_device_id = devices[random_dev]->id();
    auto src_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(src_physical_device_id);
    auto mesh_shape = control_plane.get_physical_mesh_shape(src_fabric_node_id.mesh_id);
    auto* sender_device = DevicePool::instance().get_active_device(src_physical_device_id);
    // Randomly select a tx core
    const auto& worker_grid_size = sender_device->compute_with_storage_grid_size();
    auto sender_x = std::uniform_int_distribution<uint32_t>(0, worker_grid_size.x - 2)(global_rng);
    auto sender_y = std::uniform_int_distribution<uint32_t>(0, worker_grid_size.y - 2)(global_rng);
    CoreCoord sender_logical_core = {sender_x, sender_y};
    CoreCoord receiver_logical_core = {0, 0};

    // Request randomized logical core from the receiver host
    distributed_context->recv(
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&receiver_logical_core), sizeof(receiver_logical_core)),
        tt::tt_metal::distributed::multihost::Rank{recv_host_rank},  // receive from receiver host
        tt::tt_metal::distributed::multihost::Tag{0}                 // exchange logical core over tag 0
    );
    FabricNodeId dst_fabric_node_id(MeshId{0}, 0);
    // Receive the randomized destination fabric node id from the receiver host
    distributed_context->recv(
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&dst_fabric_node_id), sizeof(dst_fabric_node_id)),
        tt::tt_metal::distributed::multihost::Rank{recv_host_rank},  // receive from receiver host
        tt::tt_metal::distributed::multihost::Tag{0}                 // exchange fabric node id over tag 0
    );

    log_debug(tt::LogTest, "Src MeshId {} ChipId {}", *(src_fabric_node_id.mesh_id), src_fabric_node_id.chip_id);
    log_debug(tt::LogTest, "Dst MeshId {} ChipId {}", *(dst_fabric_node_id.mesh_id), dst_fabric_node_id.chip_id);

    CoreCoord receiver_virtual_core = sender_device->worker_core_from_logical_core(receiver_logical_core);
    auto receiver_noc_encoding =
        tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    // test parameters
    auto worker_mem_map = generate_worker_mem_map(sender_device);
    uint32_t target_address = worker_mem_map.target_address;

    // common compile time args for sender and receiver
    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address, worker_mem_map.test_results_size_bytes, target_address};

    // Create the sender program
    auto sender_program = tt_metal::CreateProgram();
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/multihost/fabric_tests/kernels/tt_fabric_2d_unicast_tx.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args,
            .defines = {}});

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
        *dst_fabric_node_id.mesh_id};

    tt_fabric::append_fabric_connection_rt_args(
        src_fabric_node_id, dst_fabric_node_id, 0, sender_program, {sender_logical_core}, sender_runtime_args);
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Run sender program
    fixture->RunProgramNonblocking(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(sender_device, sender_program);

    // Validate status of sender
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

    // Send test results to the receiver host
    uint64_t receiver_bytes = 0;
    distributed_context->send(
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&sender_bytes), sizeof(sender_bytes)),
        tt::tt_metal::distributed::multihost::Rank{recv_host_rank},  // send to receiver host
        tt::tt_metal::distributed::multihost::Tag{0}                 // exchange tests results over tag 0
    );
    // Request test results from the receiver host and ensure that they match
    distributed_context->recv(
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&receiver_bytes), sizeof(receiver_bytes)),
        tt::tt_metal::distributed::multihost::Rank{recv_host_rank},  // recv from receiver host
        tt::tt_metal::distributed::multihost::Tag{0}                 // exchange tests results over tag 0
    );
    EXPECT_EQ(sender_bytes, receiver_bytes);
}

void run_unicast_recv_step(BaseFabricFixture* fixture, tt::tt_metal::distributed::multihost::Rank sender_host_rank) {
    // The following code runs on the receiver host
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    constexpr uint32_t num_packets = 100;
    const auto& fabric_context = control_plane.get_fabric_context();
    const auto topology = fabric_context.get_fabric_topology();
    TT_FATAL(topology == Topology::Mesh, "Intermesh Routing tests need Dynamic Routing enabled.");
    auto devices = fixture->get_devices();
    const auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();

    // Synchronize seeds across hosts (sender and receiver must use the same seed for randomization)
    uint32_t time_seed = 0;
    distributed_context->recv(
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&time_seed), sizeof(time_seed)),
        tt::tt_metal::distributed::multihost::Rank{sender_host_rank},  // recv from sender host
        tt::tt_metal::distributed::multihost::Tag{0}                   // exchange seed over tag 0
    );

    // Randomly select an rx device
    auto random_dev = std::uniform_int_distribution<uint32_t>(0, devices.size() - 1)(global_rng);
    auto dst_physical_device_id = devices[random_dev]->id();
    auto dst_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(dst_physical_device_id);
    auto* receiver_device = DevicePool::instance().get_active_device(dst_physical_device_id);

    // Randomly select an rx core
    const auto& worker_grid_size = receiver_device->compute_with_storage_grid_size();
    auto recv_x = std::uniform_int_distribution<uint32_t>(0, worker_grid_size.x - 2)(global_rng);
    auto recv_y = std::uniform_int_distribution<uint32_t>(0, worker_grid_size.y - 2)(global_rng);
    CoreCoord receiver_logical_core = {recv_x, recv_x};

    // Send the randomized rx core to the sender host, so it can send packets to the correct destination
    distributed_context->send(
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&receiver_logical_core), sizeof(receiver_logical_core)),
        tt::tt_metal::distributed::multihost::Rank{sender_host_rank},  // send to sender host
        tt::tt_metal::distributed::multihost::Tag{0}                   // exchange logical core over tag 0
    );

    // Send the randomized rx fabric node id to the sender host, so it can send packets to the correct destination
    distributed_context->send(
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&dst_fabric_node_id), sizeof(dst_fabric_node_id)),
        tt::tt_metal::distributed::multihost::Rank{sender_host_rank},  // send to sender host
        tt::tt_metal::distributed::multihost::Tag{0}                   // exchange fabric node id over tag 0
    );

    // test parameters
    auto worker_mem_map = generate_worker_mem_map(receiver_device);
    uint32_t target_address = worker_mem_map.target_address;

    // Create the receiver program
    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address, worker_mem_map.test_results_size_bytes, target_address, false};

    std::vector<uint32_t> receiver_runtime_args = {worker_mem_map.packet_payload_size_bytes, num_packets, time_seed};

    auto recv_program = create_receiver_program(compile_time_args, receiver_runtime_args, receiver_logical_core);

    // Run receiver program
    fixture->RunProgramNonblocking(receiver_device, *recv_program);
    fixture->WaitForSingleProgramDone(receiver_device, *recv_program);

    // Validate status of the receiver
    std::vector<uint32_t> receiver_status;

    tt_metal::detail::ReadFromDeviceL1(
        receiver_device,
        receiver_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        receiver_status,
        CoreType::WORKER);
    EXPECT_EQ(receiver_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
    uint64_t receiver_bytes =
        ((uint64_t)receiver_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | receiver_status[TT_FABRIC_WORD_CNT_INDEX];
    // Get test results from the sender host and ensure that they match
    uint64_t sender_bytes = 0;
    distributed_context->recv(
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&sender_bytes), sizeof(sender_bytes)),
        tt::tt_metal::distributed::multihost::Rank{sender_host_rank},  // recv from sender host
        tt::tt_metal::distributed::multihost::Tag{0}                   // exchange tests results over tag 0
    );
    // Send test results to the sender host
    distributed_context->send(
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&receiver_bytes), sizeof(receiver_bytes)),
        tt::tt_metal::distributed::multihost::Rank{sender_host_rank},  // send to sender host
        tt::tt_metal::distributed::multihost::Tag{0}                   // exchange tests results over tag 0
    );
    EXPECT_EQ(sender_bytes, receiver_bytes);
}

void run_mcast_sender_step(
    BaseFabricFixture* fixture,
    FabricNodeId mcast_sender_node,
    FabricNodeId mcast_start_node,
    const std::vector<McastRoutingInfo>& mcast_routing_info,
    const std::vector<FabricNodeId>& mcast_group_node_ids) {
    // The following code runs on the sender host
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();

    constexpr uint32_t num_packets = 100;

    const auto& fabric_context = control_plane.get_fabric_context();
    const auto topology = fabric_context.get_fabric_topology();
    TT_FATAL(topology == Topology::Mesh, "Intermesh Routing tests need Dynamic Routing enabled.");
    const auto& edm_config = fabric_context.get_fabric_router_config();

    // Synchronize seeds across hosts (sender and receiver must use the same seed for randomization)
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();
    distributed_context->send(
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&time_seed), sizeof(time_seed)),
        tt::tt_metal::distributed::multihost::Rank{1},  // send to receiver host
        tt::tt_metal::distributed::multihost::Tag{0}    // exchange seed over tag 0
    );
    // Randomly select a mcast sender device
    auto sender_phys_id = control_plane.get_physical_chip_id_from_fabric_node_id(mcast_sender_node);
    auto sender_device = DevicePool::instance().get_active_device(sender_phys_id);
    const auto& worker_grid_size = sender_device->compute_with_storage_grid_size();
    // Randomly select a mcast sender core
    auto sender_x = std::uniform_int_distribution<uint32_t>(0, worker_grid_size.x - 2)(global_rng);
    auto sender_y = std::uniform_int_distribution<uint32_t>(0, worker_grid_size.y - 2)(global_rng);
    CoreCoord sender_logical_core = {sender_x, sender_y};

    // Request randomized logical core from the receiver host
    CoreCoord receiver_logical_core = {0, 0};
    distributed_context->recv(
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&receiver_logical_core), sizeof(receiver_logical_core)),
        tt::tt_metal::distributed::multihost::Rank{1},  // receive from receiver host
        tt::tt_metal::distributed::multihost::Tag{0}    // exchange logical core over tag 0
    );

    CoreCoord receiver_virtual_core = sender_device->worker_core_from_logical_core(receiver_logical_core);
    auto receiver_noc_encoding =
        tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(receiver_virtual_core.x, receiver_virtual_core.y);

    auto worker_mem_map = generate_worker_mem_map(sender_device);
    uint32_t target_address = worker_mem_map.target_address;

    // Create the mcast sender program
    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address, worker_mem_map.test_results_size_bytes, target_address};

    auto mcast_send_program = tt_metal::CreateProgram();
    auto mcast_send_kernel = tt_metal::CreateKernel(
        mcast_send_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_line_mcast_tx.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args,
            .defines = {}});

    std::vector<uint32_t> sender_runtime_args = {
        worker_mem_map.packet_header_address,
        worker_mem_map.source_l1_buffer_address,
        worker_mem_map.packet_payload_size_bytes,
        num_packets,
        receiver_noc_encoding,
        time_seed,
        mcast_start_node.chip_id,
        *mcast_start_node.mesh_id};

    std::vector<uint32_t> mcast_header_rtas(4, 0);
    for (const auto& routing_info : mcast_routing_info) {
        mcast_header_rtas[static_cast<uint32_t>(
            control_plane.routing_direction_to_eth_direction(routing_info.mcast_dir))] = routing_info.num_mcast_hops;
    }

    sender_runtime_args.insert(sender_runtime_args.end(), mcast_header_rtas.begin(), mcast_header_rtas.end());

    log_debug(tt::LogTest, "Using edm port {} in direction {}", edm_port, edm_direction);

    tt_fabric::append_fabric_connection_rt_args(
        mcast_sender_node, mcast_start_node, 0, mcast_send_program, {sender_logical_core}, sender_runtime_args);

    tt_metal::SetRuntimeArgs(mcast_send_program, mcast_send_kernel, sender_logical_core, sender_runtime_args);

    log_debug(tt::LogTest, "Run Sender on: {}", sender_device->id());
    fixture->RunProgramNonblocking(sender_device, mcast_send_program);
    fixture->WaitForSingleProgramDone(sender_device, mcast_send_program);

    // Validate status of sender
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
    // Send test results to the receiver host
    distributed_context->send(
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&sender_bytes), sizeof(sender_bytes)),
        tt::tt_metal::distributed::multihost::Rank{1},  // send to receiver host
        tt::tt_metal::distributed::multihost::Tag{0}    // exchange test results over tag 0
    );
    // Request test results from all devices on the receiver host and ensure that they match
    for (std::size_t recv_idx = 0; recv_idx < mcast_group_node_ids.size() + 1; recv_idx++) {
        uint64_t recv_bytes = 0;
        distributed_context->recv(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&recv_bytes), sizeof(recv_bytes)),
            tt::tt_metal::distributed::multihost::Rank{1},  // recv from receiver host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange test results over tag 0
        );
        EXPECT_EQ(sender_bytes, recv_bytes);
    }
}

void run_mcast_recv_step(
    BaseFabricFixture* fixture, FabricNodeId mcast_start_node, const std::vector<FabricNodeId>& mcast_group_node_ids) {
    // The following code runs on the receiver host
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    constexpr uint32_t num_packets = 100;

    const auto& fabric_context = control_plane.get_fabric_context();
    const auto topology = fabric_context.get_fabric_topology();
    TT_FATAL(topology == Topology::Mesh, "Intermesh Routing tests need Dynamic Routing enabled.");
    const auto& edm_config = fabric_context.get_fabric_router_config();

    // Synchronize seeds across hosts (sender and receiver must use the same seed for randomization)
    uint32_t time_seed = 0;
    distributed_context->recv(
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&time_seed), sizeof(time_seed)),
        tt::tt_metal::distributed::multihost::Rank{0},  // recv from sender host
        tt::tt_metal::distributed::multihost::Tag{0}    // exchange seed over tag 0
    );
    // Query the mcast start device
    auto mcast_start_phys_id = control_plane.get_physical_chip_id_from_fabric_node_id(mcast_start_node);
    auto mcast_start_device = DevicePool::instance().get_active_device(mcast_start_phys_id);

    // Randomly select an mcast receiver core
    const auto& worker_grid_size = mcast_start_device->compute_with_storage_grid_size();
    auto recv_x = std::uniform_int_distribution<uint32_t>(0, worker_grid_size.x - 2)(global_rng);
    auto recv_y = std::uniform_int_distribution<uint32_t>(0, worker_grid_size.y - 2)(global_rng);
    CoreCoord receiver_logical_core = {recv_x, recv_x};

    // Send the randomized receiver core to the sender host, so it can send packets to the correct destination
    distributed_context->send(
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&receiver_logical_core), sizeof(receiver_logical_core)),
        tt::tt_metal::distributed::multihost::Rank{0},  // send to sender host
        tt::tt_metal::distributed::multihost::Tag{0}    // exchange logical core over tag 0
    );
    // Query the mcast group devices
    std::vector<tt_metal::IDevice*> mcast_group_devices = {};
    mcast_group_devices.reserve(mcast_group_node_ids.size());
    for (auto mcast_node_id : mcast_group_node_ids) {
        mcast_group_devices.push_back(DevicePool::instance().get_active_device(
            control_plane.get_physical_chip_id_from_fabric_node_id(mcast_node_id)));
    }
    // Test parameters
    auto worker_mem_map = generate_worker_mem_map(mcast_start_device);
    uint32_t target_address = worker_mem_map.target_address;

    // Create the mcast receiver programs
    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address, worker_mem_map.test_results_size_bytes, target_address, false};

    std::vector<uint32_t> receiver_runtime_args = {worker_mem_map.packet_payload_size_bytes, num_packets, time_seed};
    std::unordered_map<tt_metal::IDevice*, std::shared_ptr<tt_metal::Program>> recv_programs;

    recv_programs[mcast_start_device] =
        create_receiver_program(compile_time_args, receiver_runtime_args, receiver_logical_core);
    for (const auto& dev : mcast_group_devices) {
        recv_programs[dev] = create_receiver_program(compile_time_args, receiver_runtime_args, receiver_logical_core);
    }
    // Run the mcast receiver programs
    for (auto& [dev, recv_program] : recv_programs) {
        log_debug(tt::LogTest, "Run receiver on: {}", dev->id());
        fixture->RunProgramNonblocking(dev, *recv_program);
    }

    for (auto& [dev, recv_program] : recv_programs) {
        fixture->WaitForSingleProgramDone(dev, *recv_program);
    }
    // Validate status of the receiver
    // Request test results from the sender host and ensure that they match
    uint64_t sender_bytes = 0;
    distributed_context->recv(
        tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&sender_bytes), sizeof(sender_bytes)),
        tt::tt_metal::distributed::multihost::Rank{0},  // recv from sender host
        tt::tt_metal::distributed::multihost::Tag{0}    // exchange tests results over tag 0
    );

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
        // Send test results for this device to the sender host
        uint64_t receiver_bytes =
            ((uint64_t)receiver_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | receiver_status[TT_FABRIC_WORD_CNT_INDEX];
        distributed_context->send(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&receiver_bytes), sizeof(receiver_bytes)),
            tt::tt_metal::distributed::multihost::Rank{0},  // send to sender host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange test results over tag 0
        );
        EXPECT_EQ(sender_bytes, receiver_bytes);
    }
}

void RandomizedInterMeshUnicast(BaseFabricFixture* fixture) {
    auto distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    if (*(distributed_context->rank()) == 0) {
        run_unicast_sender_step(fixture, tt::tt_metal::distributed::multihost::Rank{1});
    } else {
        run_unicast_recv_step(fixture, tt::tt_metal::distributed::multihost::Rank{0});
    }
}

void InterMeshLineMcast(
    BaseFabricFixture* fixture,
    FabricNodeId mcast_sender_node,
    FabricNodeId mcast_start_node,
    const std::vector<McastRoutingInfo>& mcast_routing_info,
    const std::vector<FabricNodeId>& mcast_group_node_ids) {
    auto distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();

    if (*(distributed_context->rank()) == 0) {
        run_mcast_sender_step(fixture, mcast_sender_node, mcast_start_node, mcast_routing_info, mcast_group_node_ids);
    } else {
        run_mcast_recv_step(fixture, mcast_start_node, mcast_group_node_ids);
    }
}

std::map<FabricNodeId, chip_id_t> get_physical_chip_mapping_from_eth_coords_mapping(
    const std::vector<std::vector<eth_coord_t>>& mesh_graph_eth_coords, uint32_t local_mesh_id) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    std::map<FabricNodeId, chip_id_t> physical_chip_ids_mapping;
    for (std::uint32_t mesh_id = 0; mesh_id < mesh_graph_eth_coords.size(); mesh_id++) {
        if (mesh_id == local_mesh_id) {
            for (std::uint32_t chip_id = 0; chip_id < mesh_graph_eth_coords[mesh_id].size(); chip_id++) {
                const auto& eth_coord = mesh_graph_eth_coords[mesh_id][chip_id];
                physical_chip_ids_mapping.insert(
                    {FabricNodeId(MeshId{mesh_id}, chip_id), cluster.get_physical_chip_id_from_eth_coord(eth_coord)});
            }
        }
    }
    return physical_chip_ids_mapping;
}

}  // namespace multihost_utils

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
