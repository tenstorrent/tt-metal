// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_router_channel_mapping.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include <tt_stl/assert.hpp>

#include <vector>
namespace tt::tt_fabric {

FabricRouterChannelMapping::FabricRouterChannelMapping(
    Topology topology, eth_chan_directions direction, bool downstream_is_tensix_builder, RouterVariant variant) :
    topology_(topology), direction_(direction), downstream_is_tensix_builder_(downstream_is_tensix_builder), variant_(variant) {
    initialize_mappings();
}

void FabricRouterChannelMapping::initialize_mappings() {
    initialize_vc0_mappings();
    initialize_vc1_mappings();
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
    const bool is_2d = is_2d_topology();
    if (!is_2d) {
        // VC1 (intermesh) only exists for 2D topologies and Z routers
        return;
    }

    if (is_z_router()) {
        // Z Router VC1 layout:
        // - Sender channels 0-3: Map to erisc internal channels 4-7 (Z→mesh traffic)
        // - Receiver channel 0: Maps to erisc internal channel 1 (mesh→Z traffic)
        
        // 4 sender channels for Z router (one per potential mesh router direction)
        for (uint32_t i = 0; i < 4; ++i) {
            sender_channel_map_[LogicalSenderChannelKey{1, i}] =
                InternalSenderChannelMapping{BuilderType::ERISC, 4 + i};  // erisc channels 4-7
        }
        
        // 1 receiver channel for Z router
        receiver_channel_map_[LogicalReceiverChannelKey{1, 0}] =
            InternalReceiverChannelMapping{BuilderType::ERISC, 1};  // erisc receiver channel 1
    } else {
        // Standard mesh router VC1: [0-2] for intermesh traffic
        // Map to erisc builder channels 4-6 (after VC0 channels 0-3)
        uint32_t num_vc1_channels = 3;  // Standard 2D intermesh

        for (uint32_t i = 0; i < num_vc1_channels; ++i) {
            sender_channel_map_[LogicalSenderChannelKey{1, i}] =
                InternalSenderChannelMapping{BuilderType::ERISC, 4 + i};  // erisc channels 4-6

            receiver_channel_map_[LogicalReceiverChannelKey{1, i}] =
                InternalReceiverChannelMapping{BuilderType::ERISC, 1 + i};  // erisc receiver channels 1-3
        }
    }
}

bool FabricRouterChannelMapping::is_2d_topology() const {
    return topology_ == Topology::Mesh || topology_ == Topology::Torus;
}

bool FabricRouterChannelMapping::is_ring_or_torus() const {
    return topology_ == Topology::Ring || topology_ == Topology::Torus;
}

bool FabricRouterChannelMapping::is_z_router() const {
    return variant_ == RouterVariant::Z_ROUTER;
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
    // Z routers have 2 VCs: VC0 (mesh traffic) and VC1 (Z traffic)
    if (is_z_router()) {
        return 2;
    }
    
    // Standard mesh routers: VC0 only for now
    // TODO: Enable VC1 for standard mesh routers when intermesh support is fully implemented
    return 1;  // VC0 only
}

uint32_t FabricRouterChannelMapping::get_num_sender_channels_for_vc(uint32_t vc) const {
    switch (vc) {
        case 0:  // VC0
            return is_2d_topology() ? 4 : 2;
        case 1:  // VC1
            if (is_z_router()) {
                return 4;  // Z router has 4 sender channels (one per mesh router direction)
            } else if (is_2d_topology()) {
                return 3;  // Standard mesh router has 3 intermesh channels
            }
            return 0;  // 1D topologies don't have VC1
        default:
            return 0;
    }
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
