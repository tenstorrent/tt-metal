// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h"
#include "tt_metal/fabric/builder/mesh_channel_spec.hpp"

#include <vector>

namespace tt::tt_fabric {

// Forward declaration
struct IntermeshVCConfig;

enum class BuilderType : uint8_t {
    ERISC = 0,
    TENSIX = 1,
};

/**
 * RouterVariant - Distinguishes between mesh and Z routers
 *
 * MESH: Standard mesh router (N/E/S/W directions)
 * Z_ROUTER: Vertical Z router for inter-device connectivity
 */
enum class RouterVariant : uint8_t {
    MESH = 0,
    Z_ROUTER = 1,
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
 * This mapping is computed based on topology, direction, router variant, and tensix extension mode.
 *
 * Channel indices are relative to each VC:
 * - VC0 (1D): [0] = local worker, [1] = forwarding from upstream
 * - VC0 (2D): [0] = local worker, [1-3] = forwarding from upstream routers
 * - VC1 (2D): [0-2] = intermesh channels (standard 2D)
 * - VC1 (Z router): [0-3] = Z→mesh channels (4 sender channels mapping to 2-4 mesh routers)
 */
class FabricRouterChannelMapping {
public:
    FabricRouterChannelMapping(
        Topology topology, const MeshChannelSpec& spec, bool has_tensix_extension, RouterVariant variant);

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

    std::vector<InternalSenderChannelMapping> get_all_sender_mappings(const MeshChannelSpec& spec) const;

    /**
     * Check if this is a Z router
     */
    bool is_z_router() const;

private:
    Topology topology_;
    bool downstream_is_tensix_builder_;
    RouterVariant variant_;

    std::map<LogicalSenderChannelKey, InternalSenderChannelMapping> sender_channel_map_;
    std::map<LogicalReceiverChannelKey, InternalReceiverChannelMapping> receiver_channel_map_;

    void initialize_mappings(const MeshChannelSpec& spec);
    void initialize_vc0_mappings();
    void initialize_vc1_mappings(const MeshChannelSpec& spec);
};

}  // namespace tt::tt_fabric
