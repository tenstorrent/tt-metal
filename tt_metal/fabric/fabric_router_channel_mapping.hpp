// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h"

#include <vector>

namespace tt::tt_fabric {

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
 * This mapping is computed based on topology, direction, and tensix extension mode.
 *
 * Channel indices are relative to each VC:
 * - VC0 (1D): [0] = local worker, [1] = forwarding from upstream
 * - VC0 (2D): [0] = local worker, [1-3] = forwarding from upstream routers
 * - VC1: [0-2] or [0-3] = intermesh channels (2D/2D+Z only)
 */
class FabricRouterChannelMapping {
public:
    FabricRouterChannelMapping(
        Topology topology,
        eth_chan_directions direction,
        bool has_tensix_extension = false);

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

    std::vector<InternalSenderChannelMapping> get_all_sender_mappings() const;

private:
    Topology topology_;
    // will become used when Z-link support is added
    [[maybe_unused]] eth_chan_directions direction_;
    bool downstream_is_tensix_builder_;

    std::map<LogicalSenderChannelKey, InternalSenderChannelMapping> sender_channel_map_;
    std::map<LogicalReceiverChannelKey, InternalReceiverChannelMapping> receiver_channel_map_;

    void initialize_mappings();
    void initialize_vc0_mappings();
    void initialize_vc1_mappings();
    bool is_2d_topology() const;
    bool is_ring_or_torus() const;
};

}  // namespace tt::tt_fabric
