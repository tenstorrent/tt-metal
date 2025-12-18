// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_router_channel_mapping.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include <tt_stl/assert.hpp>

#include <vector>
namespace tt::tt_fabric {

FabricRouterChannelMapping::FabricRouterChannelMapping(
    Topology topology, const MeshChannelSpec& spec, bool downstream_is_tensix_builder, RouterVariant variant) :
    topology_(topology), downstream_is_tensix_builder_(downstream_is_tensix_builder), variant_(variant) {
    initialize_mappings(spec);
}

void FabricRouterChannelMapping::initialize_mappings(const MeshChannelSpec& spec) {
    initialize_vc0_mappings();
    initialize_vc1_mappings(spec);
}

void FabricRouterChannelMapping::initialize_vc0_mappings() {
    const bool is_2d = is_2D_topology(topology_);

    if (is_2d) {
        // 2D topology VC0 has 4 sender channels (relative indices within VC0):
        //   [0] = local worker channel
        //   [1-3] = forwarding channels from upstream routers
        // The mapping of which upstream router uses which channel depends on the downstream router's direction
        for (uint32_t i = 0; i < builder_config::num_sender_channels_2d_mesh; ++i) {
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
        sender_channel_map_[LogicalSenderChannelKey{0, 0}] =
            InternalSenderChannelMapping{vc0_builder_type, 0};  // worker channel
        sender_channel_map_[LogicalSenderChannelKey{0, 1}] =
            InternalSenderChannelMapping{vc0_builder_type, 1};  // forward channel
    }
    // Receiver channel (typically single receiver channel per VC)
    receiver_channel_map_[LogicalReceiverChannelKey{0, 0}] = InternalReceiverChannelMapping{BuilderType::ERISC, 0};
}

void FabricRouterChannelMapping::initialize_vc1_mappings(const MeshChannelSpec& spec) {
    const bool is_2d = is_2D_topology(topology_);
    if (!is_2d) {
        // VC1 (intermesh) only exists for 2D topologies and Z routers
        return;
    }

    if (is_z_router()) {
        // Z routers exist only for intermesh connectivity
        TT_FATAL(spec.has_vc(1), "Z router requires VC1 for Z traffic");
        TT_FATAL(
            spec.get_z_router_sender_channel_count_for_vc(0) == builder_config::num_sender_channels_z_router_vc0,
            "Z router VC0 sender channel count mismatch: spec has {}, expected {}",
            spec.get_z_router_sender_channel_count_for_vc(0),
            builder_config::num_sender_channels_z_router_vc0);
        TT_FATAL(
            spec.get_z_router_sender_channel_count_for_vc(1) == builder_config::num_sender_channels_z_router_vc1,
            "Z router VC1 sender channel count mismatch: spec has {}, expected {}",
            spec.get_z_router_sender_channel_count_for_vc(1),
            builder_config::num_sender_channels_z_router_vc1);

        // Z Router VC1 layout:
        // - Sender channels 0-3: Map to erisc internal channels 4-7 (Z→mesh traffic)
        // - Receiver channel 0: Maps to erisc internal channel 1 (mesh→Z traffic)

        constexpr uint32_t z_router_vc1_base_sender_channel = builder_config::num_sender_channels_z_router_vc0;
        constexpr uint32_t z_router_vc1_receiver_channel = 1;

        for (uint32_t i = 0; i < spec.get_z_router_sender_channel_count_for_vc(1); ++i) {
            sender_channel_map_[LogicalSenderChannelKey{1, i}] =
                InternalSenderChannelMapping{BuilderType::ERISC, z_router_vc1_base_sender_channel + i};
        }

        receiver_channel_map_[LogicalReceiverChannelKey{1, 0}] =
            InternalReceiverChannelMapping{BuilderType::ERISC, z_router_vc1_receiver_channel};
    } else {
        // Standard mesh router VC1: create if spec has VC1
        if (spec.has_vc(1)) {
            constexpr uint32_t mesh_vc1_base_sender_channel = builder_config::num_sender_channels_2d_mesh;
            constexpr uint32_t mesh_vc1_receiver_channel = 1;

            // Create sender channels from spec
            for (uint32_t i = 0; i < spec.get_sender_channel_count_for_vc(1); ++i) {
                sender_channel_map_[LogicalSenderChannelKey{1, i}] =
                    InternalSenderChannelMapping{BuilderType::ERISC, mesh_vc1_base_sender_channel + i};
            }

            // Create ONE receiver channel for VC1
            // A receiver channel forwards to multiple downstream sender channels
            receiver_channel_map_[LogicalReceiverChannelKey{1, 0}] =
                InternalReceiverChannelMapping{BuilderType::ERISC, mesh_vc1_receiver_channel};
        }
        // If VC1 not present in spec, don't create VC1 mappings
    }
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

std::vector<InternalSenderChannelMapping> FabricRouterChannelMapping::get_all_sender_mappings(
    const MeshChannelSpec& spec) const {
    std::vector<InternalSenderChannelMapping> result;

    // Iterate through VCs in order and flatten
    for (uint32_t vc = 0; vc < spec.get_num_vcs(); ++vc) {
        // Use the appropriate channel count based on router variant
        size_t num_channels = (variant_ == RouterVariant::Z_ROUTER) ? spec.get_z_router_sender_channel_count_for_vc(vc)
                                                                    : spec.get_sender_channel_count_for_vc(vc);

        for (uint32_t ch_idx = 0; ch_idx < num_channels; ++ch_idx) {
            result.push_back(get_sender_mapping(vc, ch_idx));
        }
    }

    return result;
}

}  // namespace tt::tt_fabric
