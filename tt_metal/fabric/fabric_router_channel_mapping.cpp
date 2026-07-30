// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_router_channel_mapping.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"
#include "tt_metal/fabric/builder/router_connection_mapping.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include <tt_stl/assert.hpp>

#include <vector>
namespace tt::tt_fabric {

FabricRouterChannelMapping::FabricRouterChannelMapping(
    Topology topology,
    bool downstream_is_tensix_builder,
    RoutingDirection direction,
    EdgeCapability edge_capability,
    const IntermeshVCConfig* intermesh_config,
    bool has_intermesh_z_edge,
    bool express_routing_enabled) :
    topology_(topology),
    downstream_is_tensix_builder_(downstream_is_tensix_builder),
    shape_(RouterConnectionMapping::router_vc_shape(
        topology, direction, edge_capability, has_intermesh_z_edge, express_routing_enabled, intermesh_config)) {
    // Logged here rather than during layout: these are the inputs to the shape derivation, and this
    // is the only point at which they are in scope. None is retained.
    log_debug(
        LogFabric,
        "FabricRouterChannelMapping: direction={}, capability={}, intermesh_z_edge={}, express={}, senders per VC = "
        "{}/{}/{}",
        static_cast<int>(direction),
        to_string(edge_capability),
        has_intermesh_z_edge,
        express_routing_enabled,
        shape_.sender_counts[0],
        shape_.sender_counts[1],
        shape_.sender_counts[2]);

    initialize_mappings();
}

void FabricRouterChannelMapping::initialize_mappings() {
    initialize_vc0_mappings();
    initialize_vc1_mappings();
    initialize_vc2_mappings();
}

void FabricRouterChannelMapping::initialize_vc0_mappings() {
    const bool is_2d = is_2D_topology(topology_);

    if (is_2d) {
        // 2D topology VC0 sender channels. The count comes from the per-VC shape computed at
        // construction: 5 for the Z-facing boundary and express families, 4 for legacy mesh, and
        // everything downstream (bases, stream assignment, CT args) reads the same fact.
        const auto num_sender_channels = shape_.sender_counts[0];

        for (uint32_t i = 0; i < num_sender_channels; ++i) {
            // When mux extension is enabled, ALL VC0 channels go to TENSIX mux
            BuilderType builder_type = downstream_is_tensix_builder_ ? BuilderType::TENSIX : BuilderType::ERISC;
            sender_channel_map_[LogicalSenderChannelKey{0, i}] =
                InternalSenderChannelMapping{builder_type, i};
        }
    } else if (topology_ == Topology::NeighborExchange) {
        // Neighbor Exchange topology VC0 has 1 sender channel:
        //  [0] = local worker channel
        // Neighbor Exchange topology currently does not support mux extension
        TT_FATAL(!downstream_is_tensix_builder_, "Neighbor Exchange topology does not support mux extension");
        sender_channel_map_[LogicalSenderChannelKey{0, 0}] = InternalSenderChannelMapping{BuilderType::ERISC, 0};
    } else {
        // 1D topology VC0 has 2 sender channels (relative indices within VC0):
        //   [0] = local worker channel
        //   [1] = forwarding channel from upstream router
        // When mux extension is enabled, ALL VC0 channels go to TENSIX mux
        BuilderType vc0_builder_type = downstream_is_tensix_builder_ ? BuilderType::TENSIX : BuilderType::ERISC;
        for (uint32_t i = 0; i < shape_.sender_counts[0]; ++i) {
            sender_channel_map_[LogicalSenderChannelKey{0, i}] = InternalSenderChannelMapping{vc0_builder_type, i};
        }
    }
    // Receiver channel (typically single receiver channel per VC)
    receiver_channel_map_[LogicalReceiverChannelKey{0, 0}] = InternalReceiverChannelMapping{BuilderType::ERISC, 0};
}

void FabricRouterChannelMapping::initialize_vc1_mappings() {
    if (!is_2D_topology(topology_) || shape_.sender_counts[1] == 0) {
        // VC1 only exists on 2D topologies, and only when the VC is enabled. The boundary family's
        // VC1 requirement is enforced upstream, in the shape derivation.
        return;
    }

    // One layout for every family: the shape already carries the count (3 legacy, 4 for a from-Z
    // slot or express, 4 for the boundary's fanout) and the flat base as a prefix sum, so there
    // is nothing left to derive here -- only entries to place. The VC1 receiver index is the
    // receiver-side prefix sum, identical to the channel 1 both old branches used.
    const uint32_t vc1_base = shape_.sender_flat_base[1];
    for (uint32_t i = 0; i < shape_.sender_counts[1]; ++i) {
        sender_channel_map_[LogicalSenderChannelKey{1, i}] =
            InternalSenderChannelMapping{BuilderType::ERISC, vc1_base + i};
    }

    receiver_channel_map_[LogicalReceiverChannelKey{1, 0}] =
        InternalReceiverChannelMapping{BuilderType::ERISC, shape_.receiver_flat_base[1]};
}

void FabricRouterChannelMapping::initialize_vc2_mappings() {
    if (!is_2D_topology(topology_) || shape_.sender_counts[2] == 0) {
        return;  // VC2 only for 2D topologies, and only when the VC is enabled
    }

    // The single VC2 sender goes at the shape's prefix-sum base (7, 8, or 9 depending on the
    // family). The one family difference the shape already carries: mesh routers service a VC2
    // receiver, the Z-facing intermesh boundary does not.
    sender_channel_map_[LogicalSenderChannelKey{2, 0}] =
        InternalSenderChannelMapping{BuilderType::ERISC, shape_.sender_flat_base[2]};

    if (shape_.receiver_counts[2] > 0) {
        receiver_channel_map_[LogicalReceiverChannelKey{2, 0}] =
            InternalReceiverChannelMapping{BuilderType::ERISC, shape_.receiver_flat_base[2]};
    }
}

InternalSenderChannelMapping FabricRouterChannelMapping::get_sender_mapping(
    uint32_t vc, uint32_t sender_channel_idx) const {
    LogicalSenderChannelKey key{vc, sender_channel_idx};
    auto it = sender_channel_map_.find(key);
    TT_FATAL(it != sender_channel_map_.end(), "No mapping found for VC{} sender channel {}", vc, sender_channel_idx);
    return it->second;
}

InternalReceiverChannelMapping FabricRouterChannelMapping::get_receiver_mapping(
    uint32_t vc, uint32_t receiver_channel_idx) const {
    LogicalReceiverChannelKey key{vc, receiver_channel_idx};
    auto it = receiver_channel_map_.find(key);
    TT_FATAL(
        it != receiver_channel_map_.end(), "No mapping found for VC{} receiver channel {}", vc, receiver_channel_idx);
    return it->second;
}

uint32_t FabricRouterChannelMapping::get_num_virtual_channels() const {
    // Config-only by design (topology-independent), exactly as the shape derivation states it:
    // a 1D router with requires_vc1 reports 2 VCs while creating zero VC1 channels, and that
    // existing oddity is preserved rather than fixed here.
    return shape_.num_vcs;
}

uint32_t FabricRouterChannelMapping::get_num_sender_channels_for_vc(uint32_t vc) const {
    // A direct read of the shape computed at construction -- no counting of map entries.
    if (vc >= builder_config::MAX_NUM_VCS) {
        return 0;
    }
    return shape_.sender_counts[vc];
}

uint32_t FabricRouterChannelMapping::get_num_receiver_channels_for_vc(uint32_t vc) const {
    // A direct read of the shape computed at construction -- no counting of map entries.
    if (vc >= builder_config::MAX_NUM_VCS) {
        return 0;
    }
    return shape_.receiver_counts[vc];
}

std::vector<InternalSenderChannelMapping> FabricRouterChannelMapping::get_all_sender_mappings() const {
    std::vector<InternalSenderChannelMapping> result;

    // Iterate through VCs in order and flatten
    for (uint32_t vc = 0; vc < get_num_virtual_channels(); ++vc) {
        for (uint32_t ch_idx = 0; ch_idx < get_num_sender_channels_for_vc(vc); ++ch_idx) {
            result.push_back(get_sender_mapping(vc, ch_idx));
        }
    }

    return result;
}

}  // namespace tt::tt_fabric
