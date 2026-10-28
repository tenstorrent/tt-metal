// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "channel_trimming_test_utils.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <map>
#include <string>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"

#include "fabric_fixture.hpp"

namespace tt::tt_fabric::fabric_router_tests {

UnicastTrafficResult run_unicast_traffic_bw_nodes(
    FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id, uint32_t num_hops, uint32_t num_packets) {
    tt::tt_metal::CoreCoord sender_logical_core = {0, 0};
    tt::tt_metal::CoreCoord receiver_logical_core = {1, 0};

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    ChipId src_physical = control_plane.get_physical_chip_id_from_fabric_node_id(src_fabric_node_id);
    ChipId dst_physical = control_plane.get_physical_chip_id_from_fabric_node_id(dst_fabric_node_id);

    // For multi-hop in 1D mode, append_fabric_connection_rt_args only works for direct
    // neighbors. Determine the first-hop neighbor and use it for the connection setup.
    // The sender kernel uses num_hops/dst_chip_id runtime args to forward further.
    auto forwarding_dir_opt = control_plane.get_forwarding_direction(src_fabric_node_id, dst_fabric_node_id);
    if (!forwarding_dir_opt.has_value()) {
        return UnicastTrafficResult{{}, {}, FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 0), 0, true};
    }

    // The connection destination is the first-hop neighbor (which equals dst for 1-hop)
    FabricNodeId connection_dst = dst_fabric_node_id;
    if (num_hops > 1) {
        auto first_hop_neighbors =
            control_plane.get_intra_chip_neighbors(src_fabric_node_id, *forwarding_dir_opt);
        if (first_hop_neighbors.empty()) {
            return UnicastTrafficResult{{}, {}, FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 0), 0, true};
        }
        connection_dst = FabricNodeId(src_fabric_node_id.mesh_id, first_hop_neighbors[0]);
    }

    const auto& available_links = get_forwarding_link_indices(
        *BaseFabricFixture::devices_map_.at(src_physical), src_fabric_node_id, connection_dst);
    if (available_links.empty()) {
        return UnicastTrafficResult{{}, {}, FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 0), 0, true};
    }
    uint32_t link_idx = available_links[0];

    auto dir_eth_chans =
        control_plane.get_active_fabric_eth_channels_in_direction(src_fabric_node_id, *forwarding_dir_opt);
    chan_id_t edm_port = dir_eth_chans[link_idx];

    const auto& sender_device = BaseFabricFixture::devices_map_.at(src_physical);
    const auto& receiver_device = BaseFabricFixture::devices_map_.at(dst_physical);
    tt::tt_metal::CoreCoord receiver_virtual_core = receiver_device->worker_core_from_logical_core(receiver_logical_core);

    const auto topology = control_plane.get_fabric_context().get_fabric_topology();
    uint32_t is_2d_fabric = topology == Topology::Mesh || topology == Topology::Torus;

    auto worker_mem_map = BaseFabricFixture::generate_worker_mem_map(sender_device, topology);
    uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::vector<uint32_t> compile_time_args = {
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        worker_mem_map.target_address,
        0 /* use_dram_dst */,
        is_2d_fabric,
        0 /* is_chip_multicast */,
        0 /* additional_dir */};

    std::map<std::string, std::string> defines = {};
    if (is_2d_fabric) {
        defines["FABRIC_2D"] = "";
    }

    // Create sender program
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
        worker_mem_map.source_l1_buffer_address,
        worker_mem_map.packet_payload_size_bytes,
        num_packets,
        time_seed,
        receiver_virtual_core.x,
        receiver_virtual_core.y,
        num_hops,
        1 /* fwd_range */,
        dst_fabric_node_id.chip_id,
        *dst_fabric_node_id.mesh_id};

    // Use connection API to append EDM connection args (to first-hop neighbor)
    append_fabric_connection_rt_args(
        src_fabric_node_id,
        connection_dst,
        link_idx,
        sender_program,
        sender_logical_core,
        sender_runtime_args);

    tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

    // Create receiver program
    std::vector<uint32_t> receiver_runtime_args = {worker_mem_map.packet_payload_size_bytes, num_packets, time_seed};

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

    // Launch and wait. The receiver workload must outlive the Finish below; fast dispatch may still
    // prefetch its kernel binaries after LaunchProgramAsync returns.
    auto receiver_workload = tt_metal::LaunchProgramAsync(*receiver_device, std::move(receiver_program));
    tt_metal::LaunchProgram(*sender_device, std::move(sender_program));
    tt::tt_metal::distributed::Finish(receiver_device->mesh_command_queue());

    // Validate sender/receiver status
    std::vector<uint32_t> sender_status;
    std::vector<uint32_t> receiver_status;

    tt_metal::slow_dispatch::ReadFromL1(
        *sender_device,
        sender_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        sender_status,
        CoreType::WORKER);

    tt_metal::slow_dispatch::ReadFromL1(
        *receiver_device,
        receiver_logical_core,
        worker_mem_map.test_results_address,
        worker_mem_map.test_results_size_bytes,
        receiver_status,
        CoreType::WORKER);

    EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
    EXPECT_EQ(receiver_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);

    return UnicastTrafficResult{
        src_physical, dst_physical, src_fabric_node_id, dst_fabric_node_id, edm_port, false};
}

MeshPathResult find_1d_path_with_hops(uint32_t num_hops) {
    // Walk the 1D chain from chip 0 using get_forwarding_direction to determine
    // the actual forward direction at each hop. In 1D mode, get_forwarding_link_indices
    // returns links for ANY destination (all traffic goes "forward"), so we can't use
    // it to distinguish neighbors. Instead, we ask for the forwarding direction towards
    // an unvisited chip, then get the neighbor in that direction.
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto mesh_shape = control_plane.get_physical_mesh_shape(MeshId{0});
    uint32_t num_chips = mesh_shape.mesh_size();

    FabricNodeId current(MeshId{0}, 0);
    std::vector<FabricNodeId> path = {current};

    for (uint32_t hop = 0; hop < num_hops; hop++) {
        // Find the furthest unvisited chip to use as a forwarding target.
        // Using the LAST unvisited chip avoids picking a direct neighbor, which would
        // cause get_forwarding_direction to return the 2D shortest-path direction
        // instead of the 1D chain forward direction.
        FabricNodeId target(MeshId{0}, 0);
        bool found_target = false;
        for (int i = num_chips - 1; i >= 0; i--) {
            bool visited = std::any_of(path.begin(), path.end(), [&](const FabricNodeId& n) {
                return n.chip_id == static_cast<uint32_t>(i);
            });
            if (!visited) {
                target = FabricNodeId(MeshId{0}, static_cast<uint32_t>(i));
                found_target = true;
                break;
            }
        }
        if (!found_target) {
            log_warning(tt::LogTest, "find_1d_path_with_hops: no unvisited chip at hop {}", hop);
            return MeshPathResult{current, current, {}, RoutingDirection::E, false};
        }

        // Get the forward direction from current towards the target
        auto fwd_dir = control_plane.get_forwarding_direction(current, target);
        if (!fwd_dir.has_value()) {
            log_warning(
                tt::LogTest,
                "find_1d_path_with_hops: no forwarding direction from chip_id={} to chip_id={} at hop {}",
                current.chip_id,
                target.chip_id,
                hop);
            return MeshPathResult{current, current, {}, RoutingDirection::E, false};
        }

        // Get the actual neighbor in the forward direction
        auto neighbors = control_plane.get_intra_chip_neighbors(current, *fwd_dir);
        if (neighbors.empty()) {
            log_warning(
                tt::LogTest,
                "find_1d_path_with_hops: no neighbor in direction {} from chip_id={} at hop {}",
                static_cast<int>(*fwd_dir),
                current.chip_id,
                hop);
            return MeshPathResult{current, current, {}, RoutingDirection::E, false};
        }

        FabricNodeId next(current.mesh_id, neighbors[0]);
        path.push_back(next);
        current = next;
    }

    return MeshPathResult{path.front(), path.back(), path, RoutingDirection::E, true};
}

MeshPathResult find_2d_path_with_transit(bool require_deadlock_avoidance_last_hop) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& fabric_context = control_plane.get_fabric_context();
    auto mesh_shape = control_plane.get_physical_mesh_shape(MeshId{0});
    uint32_t num_chips = mesh_shape.mesh_size();

    for (uint32_t src = 0; src < num_chips; ++src) {
        for (uint32_t dst = 0; dst < num_chips; ++dst) {
            if (src == dst) {
                continue;
            }
            FabricNodeId src_node(MeshId{0}, src);
            FabricNodeId dst_node(MeshId{0}, dst);
            auto route = control_plane.get_canonical_intramesh_route(src_node, dst_node);
            auto direction = control_plane.get_forwarding_direction(src_node, dst_node);
            // We want a route with at least 3 nodes (src, transit, dst) and a valid direction
            if (route.size() < 3 || !direction.has_value()) {
                continue;
            }
            // With require_deadlock_avoidance_last_hop, only accept a route whose last hop (transit -> dst) runs on
            // an axis where the transit router keeps deadlock avoidance, i.e. a torus ring axis.
            if (require_deadlock_avoidance_last_hop) {
                const FabricNodeId& transit_node = route[route.size() - 2];
                auto last_hop_direction = control_plane.get_forwarding_direction(transit_node, dst_node);
                if (!last_hop_direction.has_value() ||
                    !fabric_context.need_deadlock_avoidance_support(
                        control_plane,
                        transit_node,
                        control_plane.routing_direction_to_eth_direction(*last_hop_direction))) {
                    continue;
                }
            }
            return MeshPathResult{src_node, dst_node, std::move(route), *direction, true};
        }
    }

    return MeshPathResult{FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 0), {}, RoutingDirection::E, false};
}

}  // namespace tt::tt_fabric::fabric_router_tests
