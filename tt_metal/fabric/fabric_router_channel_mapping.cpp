// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_router_channel_mapping.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include <tt_stl/assert.hpp>

namespace tt::tt_fabric {

FabricRouterChannelMapping::FabricRouterChannelMapping(
    Topology topology, eth_chan_directions direction, bool downstream_is_tensix_builder) :
    topology_(topology), direction_(direction), downstream_is_tensix_builder_(downstream_is_tensix_builder) {
    initialize_mappings();
}

void FabricRouterChannelMapping::initialize_mappings() {
    initialize_vc0_mappings();
    initialize_vc1_mappings();
    initialize_vc2_mappings();
}

void FabricRouterChannelMapping::initialize_vc0_mappings() {
    const bool is_2d = is_2d_topology();

    if (is_2d) {
        // 2D topology VC0 has 4 sender channels (relative indices within VC0):
        //   [0] = local worker channel
        //   [1-3] = forwarding channels from upstream routers
        // The mapping of which upstream router uses which channel depends on the downstream router's direction
        constexpr size_t max_2d_vc0_channels = 4;
        for (uint32_t i = 0; i < max_2d_vc0_channels; ++i) {
            // When mux extension is enabled, ALL VC0 channels go to TENSIX mux
            // because ERISC only has buffers for worker channel (used by VC1) and VC1 channel
            BuilderType builder_type = downstream_is_tensix_builder_ ? BuilderType::TENSIX : BuilderType::ERISC;
            sender_channel_map_[LogicalSenderChannelKey{0, i}] =
                InternalSenderChannelMapping{builder_type, i};
        }

        // Receiver channel
        receiver_channel_map_[LogicalReceiverChannelKey{0, 0}] =
            InternalReceiverChannelMapping{BuilderType::ERISC, 0};
    } else {
        // 1D topology VC0 has 2 sender channels (relative indices within VC0):
        //   [0] = local worker channel
        //   [1] = forwarding channel from upstream router
        // When mux extension is enabled, ALL VC0 channels go to TENSIX mux
        BuilderType vc0_builder_type = downstream_is_tensix_builder_ ? BuilderType::TENSIX : BuilderType::ERISC;
        sender_channel_map_[LogicalSenderChannelKey{0, 0}] =
            InternalSenderChannelMapping{vc0_builder_type, 0};  // worker channel
        sender_channel_map_[LogicalSenderChannelKey{0, 1}] =
            InternalSenderChannelMapping{vc0_builder_type, 1};  // forward channel

        // Receiver channel (typically single receiver channel per VC)
        receiver_channel_map_[LogicalReceiverChannelKey{0, 0}] =
            InternalReceiverChannelMapping{BuilderType::ERISC, 0};
    }
}

void FabricRouterChannelMapping::initialize_vc1_mappings() {
    const bool has_ring_or_torus = is_ring_or_torus();
    if (!has_ring_or_torus) {
        // VC1 only exists for Ring/Torus topologies
        return;
    }

    const bool is_2d = is_2d_topology();
    // VC1: [0] = vc1 forward channel
    // Maps to erisc builder's last sender channel
    uint32_t vc1_sender_channel = is_2d ? builder_config::num_sender_channels_2d_torus - 1
                                         : builder_config::num_sender_channels_1d_ring - 1;

    sender_channel_map_[LogicalSenderChannelKey{1, 0}] =
        InternalSenderChannelMapping{BuilderType::ERISC, vc1_sender_channel};

    // Receiver channel for VC1
    receiver_channel_map_[LogicalReceiverChannelKey{1, 0}] =
        InternalReceiverChannelMapping{BuilderType::ERISC, 1};  // VC1 uses receiver channel 1
}

void FabricRouterChannelMapping::initialize_vc2_mappings() {
    const bool is_2d = is_2d_topology();
    if (!is_2d) {
        // VC2 (intermesh) only exists for 2D topologies
        return;
    }

    // VC2: [0-2] for 2D, [0-3] for 2D+Z
    // For now, we'll map to erisc/tensix builder channels
    // The exact mapping depends on intermesh implementation details
    // This is a placeholder - actual implementation may vary
    uint32_t num_vc2_channels = 3;  // Default for 2D, could be 4 for 2D+Z

    for (uint32_t i = 0; i < num_vc2_channels; ++i) {
        // Map to erisc builder for now - tensix mapping would be added if needed
        sender_channel_map_[LogicalSenderChannelKey{2, i}] =
            InternalSenderChannelMapping{BuilderType::ERISC, i};

        receiver_channel_map_[LogicalReceiverChannelKey{2, i}] =
            InternalReceiverChannelMapping{BuilderType::ERISC, i};
    }
}

bool FabricRouterChannelMapping::is_2d_topology() const {
    return topology_ == Topology::Mesh || topology_ == Topology::Torus;
}

bool FabricRouterChannelMapping::is_ring_or_torus() const {
    return topology_ == Topology::Ring || topology_ == Topology::Torus;
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

}  // namespace tt::tt_fabric
