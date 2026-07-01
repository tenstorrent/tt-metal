// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// FABRIC_1D per-page sparse multicast: one payload is delivered to a set of non-contiguous
// colinear chips (selected by a hop bitmask), each writing chip landing the payload at its OWN
// address (target_base + slot * payload). The 0b101 case (write hop 1, skip hop 2, write hop 3)
// validates that the router advances write_idx only on writing hops.

#include <gtest/gtest.h>
#include <bit>
#include <chrono>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "fabric_fixture.hpp"
#include "utils.hpp"

namespace tt::tt_fabric::fabric_router_tests {

// NOC_UNICAST_WRITE == 0: the receiver only checks payload correctness, independent of how it arrived.
constexpr uint32_t RECEIVER_NOC_UNICAST_WRITE = 0;

void RunSparseMcastPerPage(BaseFabricFixture* fixture, RoutingDirection dir, uint16_t hop_mask) {
    CoreCoord sender_logical_core = {0, 0};
    CoreCoord receiver_logical_core = {1, 0};
    uint32_t num_packets = 1;  // per-dest offsets are base + slot*payload; >1 would overlap streams
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto topology = control_plane.get_fabric_context().get_fabric_topology();

    FabricNodeId src_fabric_node_id(MeshId{0}, 0);
    std::unordered_map<RoutingDirection, uint32_t> fabric_hops;
    fabric_hops[dir] = std::bit_width(hop_mask);  // reach out to the most distant set hop

    std::unordered_map<RoutingDirection, std::vector<FabricNodeId>> end_fabric_node_ids_by_dir;
    ChipId src_physical_device_id;
    std::unordered_map<RoutingDirection, std::vector<ChipId>> physical_end_device_ids_by_dir;
    if (!find_device_with_neighbor_in_multi_direction(
            fixture,
            src_fabric_node_id,
            end_fabric_node_ids_by_dir,
            src_physical_device_id,
            physical_end_device_ids_by_dir,
            fabric_hops)) {
        GTEST_SKIP() << "No line of chips found for the requested direction/hop count";
    }

    // Connect only to the first hop; sparse routing to further hops is carried by the packet header.
    std::vector<FabricNodeId> dest_fabric_node_ids;
    ASSERT_FALSE(end_fabric_node_ids_by_dir[dir].empty());
    dest_fabric_node_ids.push_back(end_fabric_node_ids_by_dir[dir][0]);

    // Writing chips = set bits of hop_mask, in ascending hop order (== write_idx slot order).
    std::vector<ChipId> writing_physical_devices;
    for (size_t i = 0; i < physical_end_device_ids_by_dir[dir].size(); i++) {
        if (hop_mask & (1u << i)) {
            writing_physical_devices.push_back(physical_end_device_ids_by_dir[dir][i]);
        }
    }
    uint32_t num_dests = static_cast<uint32_t>(writing_physical_devices.size());
    ASSERT_GT(num_dests, 0u);
    ASSERT_LE(num_dests, static_cast<uint32_t>(NOC_SPARSE_MCAST_WRITE_MAX_DESTS));

    auto sender_device = fixture->get_device(src_physical_device_id);
    CoreCoord receiver_virtual_core = sender_device->worker_core_from_logical_core(receiver_logical_core);
    auto worker_mem_map = BaseFabricFixture::generate_worker_mem_map(sender_device, topology);
    uint32_t payload = worker_mem_map.packet_payload_size_bytes;

    tt_metal::Program sender_program = tt_metal::CreateProgram();
    std::vector<uint32_t> sender_ct_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        worker_mem_map.target_address,  // target_base
        num_dests};
    std::vector<uint32_t> sender_rt_args = {
        worker_mem_map.source_l1_buffer_address,
        payload,
        num_packets,
        time_seed,
        static_cast<uint32_t>(receiver_virtual_core.x),
        static_cast<uint32_t>(receiver_virtual_core.y),
        static_cast<uint32_t>(hop_mask)};
    auto sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_sparse_mcast_perpage_sender.cpp",
        {sender_logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = sender_ct_args});
    append_routing_plane_connection_manager_rt_args(
        src_fabric_node_id,
        dest_fabric_node_ids,
        {},
        sender_program,
        sender_kernel,
        {sender_logical_core},
        sender_rt_args);
    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_rt_args);

    // One receiver per writing chip; each polls its own offset (base + slot*payload).
    std::vector<std::pair<std::shared_ptr<tt_metal::distributed::MeshDevice>, tt_metal::Program>> receiver_programs;
    std::vector<uint32_t> receiver_rt_args = {payload, num_packets, time_seed};
    for (uint32_t slot = 0; slot < num_dests; slot++) {
        auto receiver_device = fixture->get_device(writing_physical_devices[slot]);
        tt_metal::Program receiver_program = tt_metal::CreateProgram();
        std::vector<uint32_t> receiver_ct_args = {
            worker_mem_map.test_results_address,
            worker_mem_map.test_results_size_bytes,
            worker_mem_map.notification_mailbox_address,
            worker_mem_map.target_address + slot * payload,  // distinct per-dest landing offset
            RECEIVER_NOC_UNICAST_WRITE,
            1,   // num_send_dir
            0,   // with_state
            1};  // is_chip_multicast
        auto receiver_kernel = tt_metal::CreateKernel(
            receiver_program,
            "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/test_linear_api_receiver.cpp",
            {receiver_logical_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = receiver_ct_args});
        tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_rt_args);
        fixture->RunProgramNonblocking(receiver_device, receiver_program);
        receiver_programs.emplace_back(receiver_device, std::move(receiver_program));
    }

    fixture->RunProgramNonblocking(sender_device, sender_program);
    fixture->WaitForSingleProgramDone(sender_device, sender_program);
    for (auto& [dev, prog] : receiver_programs) {
        fixture->WaitForSingleProgramDone(dev, prog);
    }

    std::vector<uint32_t> sender_status;
    tt_metal::detail::ReadFromDeviceL1(
        sender_device->get_devices()[0],
        sender_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        sender_status,
        CoreType::WORKER);
    EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);

    for (auto& [dev, _] : receiver_programs) {
        std::vector<uint32_t> recv_status;
        tt_metal::detail::ReadFromDeviceL1(
            dev->get_devices()[0],
            receiver_logical_core,
            worker_mem_map.test_results_address,
            worker_mem_map.test_results_size_bytes,
            recv_status,
            CoreType::WORKER);
        EXPECT_EQ(recv_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
    }
}

TEST_F(Fabric1DFixture, TestSparseMcastPerPageSingleDest) { RunSparseMcastPerPage(this, RoutingDirection::E, 0b1); }
TEST_F(Fabric1DFixture, TestSparseMcastPerPageTwoContiguous) { RunSparseMcastPerPage(this, RoutingDirection::E, 0b11); }
TEST_F(Fabric1DFixture, TestSparseMcastPerPageNonContiguous) {
    RunSparseMcastPerPage(this, RoutingDirection::E, 0b101);
}

}  // namespace tt::tt_fabric::fabric_router_tests
