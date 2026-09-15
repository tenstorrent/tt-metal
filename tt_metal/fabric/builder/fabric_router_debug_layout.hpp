// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/fabric_stream_assignment.hpp"

namespace tt::tt_fabric {

class FabricEriscDatamoverBuilder;
struct RouterLocation;

// Different types of underlying locations that a FabricRouterDebugRegion can be backed by.
enum class DebugRegionBacking : uint8_t { GROUP, UNRESERVED_L1, FIXED_L1, STREAM_REG };

// Writers that can write to a particular FabricRouterDebugRegion.
enum class DebugRegionWriter : uint8_t { NONE, ERISC0, ERISC1, ANY_ERISC, HOST, PEER, WORKER };

// A particular region of memory in the fabric router that has some purpose (ie. routing table, stream registers, etc).
struct FabricRouterDebugRegion {
    std::string id;
    std::string parent;
    DebugRegionBacking backing = DebugRegionBacking::GROUP;
    uint32_t address = 0;
    uint32_t size = 0;
    std::optional<uint32_t> count;
    std::optional<uint32_t> stride;
    std::optional<uint32_t> stream_id;
    bool allocated = true;
    bool enabled = true;
    DebugRegionWriter writer = DebugRegionWriter::NONE;
    std::string schema;
    std::vector<std::string> overlaps;

    bool operator==(const FabricRouterDebugRegion&) const = default;
};

// A collection of FabricRouterDebugRegions that describe the layout of a fabric router.
struct FabricRouterDebugLayout {
    std::vector<FabricRouterDebugRegion> regions;

    bool operator==(const FabricRouterDebugLayout&) const = default;
};

// Information about a particular fabric router instance, detailing reserved memory regions, virtual channnel
// layout, and other relevant information.
struct FabricRouterDebugInstance {
    FabricNodeId local_node = FabricNodeId(MeshId{0}, 0);
    uint32_t eth_chan = 0;
    FabricNodeId peer_node = FabricNodeId(MeshId{0}, 0);
    RoutingDirection direction = RoutingDirection::N;
    bool is_inter_mesh = false;
    bool is_dispatch_link = false;
    uint32_t num_active_eriscs = 0;
    std::array<uint32_t, builder_config::MAX_NUM_VCS> sender_channels_per_vc = {};
    std::array<uint32_t, builder_config::MAX_NUM_VCS> receiver_channels_per_vc = {};
    uint32_t worker_sender_channel = 0;
    CreditTransportPlan credit_plan;
    bool first_level_ack_vc0 = false;
    uint32_t downstream_edm_mask_vc0 = 0;
    uint32_t downstream_edm_mask_vc1 = 0;
    bool has_tensix_extension = false;
    bool udm_mode = false;
    FabricRouterDebugLayout layout;
};

// Build a FabricRouterDebugInstance for a particular fabric router instance.
FabricRouterDebugInstance build_router_debug_instance(
    const FabricEriscDatamoverBuilder& builder,
    const StreamAssignment& streams,
    const std::vector<std::unordered_map<std::string, uint32_t>>& named_ct_args_per_risc,
    const RouterLocation& location);

}  // namespace tt::tt_fabric
