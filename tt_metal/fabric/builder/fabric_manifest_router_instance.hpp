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

// Storage that backs a ManifestRouterRegion.
// Group can be thought of as the root node for a tree of regions, where
// the child nodes are the actual regions with addresses and the root simply
// acts as a container for the child nodes.
enum class ManifestRegionBacking : uint8_t { GROUP, UNRESERVED_L1, FIXED_L1, STREAM_REG };

// Who writes a ManifestRouterRegion.
enum class ManifestRegionWriter : uint8_t { NONE, ERISC0, ERISC1, ANY_ERISC, HOST, PEER, WORKER };

// One capturable slice of a router, ie. an L1 range, a fixed HAL region, a stream register, or a grouping node.
struct ManifestRouterRegion {
    std::string id;
    std::string parent;
    ManifestRegionBacking backing = ManifestRegionBacking::GROUP;
    uint32_t address = 0;
    uint32_t size = 0;
    std::optional<uint32_t> count;
    std::optional<uint32_t> stride;
    std::optional<uint32_t> stream_id;
    bool allocated = true;
    bool enabled = true;
    ManifestRegionWriter writer = ManifestRegionWriter::NONE;
    std::string schema;
    std::vector<std::string> overlaps;

    bool operator==(const ManifestRouterRegion&) const = default;
};

// One downstream connection, from a receiver channel to a sender channel, on a VC.
struct ManifestDownstreamEdge {
    // The 1-based compact slot (EDGE_1..EDGE_4).
    uint32_t edge = 0;
    eth_chan_directions direction = eth_chan_directions::EAST;
    uint32_t sender_channel = 0;
};

// "E" / "W" / "N" / "S" / "Z" for a manifest direction.
std::string direction_to_str(eth_chan_directions direction);

// A collection of ManifestRouterRegions that describe the regions of a fabric router.
struct ManifestRouterRegionLayout {
    std::vector<ManifestRouterRegion> regions;

    bool operator==(const ManifestRouterRegionLayout&) const = default;
};

// Information about a particular fabric router instance, detailing reserved memory regions, virtual channnel
// layout, and other relevant information.
struct ManifestRouterInstance {
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
    std::vector<std::optional<std::string>> sender_producers;
    CreditTransportPlan credit_plan;
    bool first_level_ack_vc0 = false;
    uint32_t downstream_edm_mask_vc0 = 0;
    uint32_t downstream_edm_mask_vc1 = 0;
    std::vector<ManifestDownstreamEdge> downstream_edges_vc0;
    std::vector<ManifestDownstreamEdge> downstream_edges_vc1;
    bool has_tensix_extension = false;
    bool udm_mode = false;
    ManifestRouterRegionLayout layout;
};

// Build a ManifestRouterInstance from a finalized router builder.
ManifestRouterInstance build_manifest_router_instance(
    const FabricEriscDatamoverBuilder& builder,
    const StreamAssignment& streams,
    const std::vector<std::unordered_map<std::string, uint32_t>>& named_ct_args_per_risc,
    const RouterLocation& location);

}  // namespace tt::tt_fabric
