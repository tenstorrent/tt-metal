// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include "tt_metal/fabric/builder/router_connection_mapping.hpp"
#include "tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h"

#include <vector>

namespace tt::tt_fabric {

// Forward declaration
struct IntermeshVCConfig;

enum class BuilderType : uint8_t {
    ERISC = 0,
    TENSIX = 1,
};

struct LogicalSenderChannelKey {
    uint32_t vc;
    uint32_t sender_channel_idx;

    bool operator<(const LogicalSenderChannelKey& other) const {
        if (vc != other.vc) {
            return vc < other.vc;
        }
        return sender_channel_idx < other.sender_channel_idx;
    }
};

struct LogicalReceiverChannelKey {
    uint32_t vc;
    uint32_t receiver_channel_idx;

    bool operator<(const LogicalReceiverChannelKey& other) const {
        if (vc != other.vc) {
            return vc < other.vc;
        }
        return receiver_channel_idx < other.receiver_channel_idx;
    }
};

struct InternalSenderChannelMapping {
    BuilderType builder_type;
    uint32_t internal_sender_channel_id;
};

struct InternalReceiverChannelMapping {
    BuilderType builder_type;
    uint32_t internal_receiver_channel_id;
};

/**
 * FabricRouterChannelMapping
 *
 * Defines the mapping from logical channels (VC + relative channel index within VC) to internal builder channels.
 * This is layout only: how many channels each VC has, and where each VC starts in the flat index
 * space, are decided once by RouterConnectionMapping::router_vc_shape() next to the wiring rules
 * that produce them. This class places entries at the indices that shape hands it and answers
 * counts by reading it back. It classifies nothing and re-derives no count.
 *
 * Channel indices are relative to each VC:
 * - VC0 (1D): [0] = local worker, [1] = forwarding from upstream
 * - VC0 (2D): [0] = local worker, [1-4] = forwarding from upstream routers
 * - VC1 (2D): [0-3] = intermesh channels
 */
class FabricRouterChannelMapping {
public:
    FabricRouterChannelMapping(
        Topology topology,
        bool downstream_is_tensix_builder,
        RoutingDirection direction,
        EdgeCapability edge_capability,
        const IntermeshVCConfig* intermesh_config,
        bool has_intermesh_z_edge = false,
        bool express_routing_enabled = false);

    /**
     * Get the internal sender channel mapping for a logical sender channel
     */
    InternalSenderChannelMapping get_sender_mapping(uint32_t vc, uint32_t sender_channel_idx) const;

    /**
     * Get the internal receiver channel mapping for a logical receiver channel
     */
    InternalReceiverChannelMapping get_receiver_mapping(uint32_t vc, uint32_t receiver_channel_idx) const;

    /**
     * Get the topology for this router
     */
    Topology get_topology() const { return topology_; }

    uint32_t get_num_virtual_channels() const;

    uint32_t get_num_sender_channels_for_vc(uint32_t vc) const;
    uint32_t get_num_receiver_channels_for_vc(uint32_t vc) const;

    std::vector<InternalSenderChannelMapping> get_all_sender_mappings() const;

private:
    Topology topology_;
    bool downstream_is_tensix_builder_;

    // The per-VC shape, computed once at construction from the same facts the connection map
    // reads. The router's facing direction, its edge capability, its chip's intermesh Z edge, and
    // the mesh's express state are inputs to that derivation only -- they are deliberately not
    // retained, so nothing here can re-answer a question the shape already settled.
    RouterConnectionMapping::RouterVcShape shape_;

    std::map<LogicalSenderChannelKey, InternalSenderChannelMapping> sender_channel_map_;
    std::map<LogicalReceiverChannelKey, InternalReceiverChannelMapping> receiver_channel_map_;

    void initialize_mappings();
    void initialize_vc0_mappings();
    void initialize_vc1_mappings();
    void initialize_vc2_mappings();
};

}  // namespace tt::tt_fabric
