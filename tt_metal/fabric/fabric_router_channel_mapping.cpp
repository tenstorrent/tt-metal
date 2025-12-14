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
    Topology topology,
    bool downstream_is_tensix_builder,
    RouterVariant variant,
    const IntermeshVCConfig* intermesh_config,
    bool is_inter_mesh_router) :
    topology_(topology),
    downstream_is_tensix_builder_(downstream_is_tensix_builder),
    variant_(variant),
    intermesh_vc_config_(intermesh_config),
    is_inter_mesh_router_(is_inter_mesh_router) {
    initialize_mappings();
}

void FabricRouterChannelMapping::initialize_mappings() {
    initialize_vc0_mappings();
    initialize_vc1_mappings();
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

void FabricRouterChannelMapping::initialize_vc1_mappings() {
    const bool is_2d = is_2D_topology(topology_);
    if (!is_2d) {
        // VC1 (intermesh) only exists for 2D topologies and Z routers
        return;
    }

    if (is_z_router()) {
        // Z routers exist only for intermesh connectivity - validate config
        TT_FATAL(
            intermesh_vc_config_ != nullptr,
            "Z router requires intermesh VC config (Z routers only exist for intermesh connectivity)");
        TT_FATAL(
            intermesh_vc_config_->requires_vc1,
            "Z router requires intermesh VC to be enabled (requires_vc1 must be true)");

        // Z Router VC1 layout:
        // - Sender channels 0-3: Map to erisc internal channels 4-7 (Z→mesh traffic)
        // - Receiver channel 0: Maps to erisc internal channel 1 (mesh→Z traffic)

        constexpr uint32_t z_router_vc1_sender_count = builder_config::num_sender_channels_z_router_vc1;
        constexpr uint32_t z_router_vc1_base_sender_channel = builder_config::num_sender_channels_z_router_vc0;
        constexpr uint32_t z_router_vc1_receiver_channel = 1;

        for (uint32_t i = 0; i < z_router_vc1_sender_count; ++i) {
            sender_channel_map_[LogicalSenderChannelKey{1, i}] =
                InternalSenderChannelMapping{BuilderType::ERISC, z_router_vc1_base_sender_channel + i};
        }

        receiver_channel_map_[LogicalReceiverChannelKey{1, 0}] =
            InternalReceiverChannelMapping{BuilderType::ERISC, z_router_vc1_receiver_channel};
    } else {
        // Standard mesh router VC1: only create if intermesh VC is required
        // Inter-mesh routers don't have VC1 (they only use VC0)
        // Intra-mesh routers have VC1 to forward inter-mesh traffic
        if (intermesh_vc_config_ && intermesh_vc_config_->requires_vc1 && !is_inter_mesh_router_) {
            // Determine sender count based on intermesh router type
            // XY intermesh: 3 sender channels (mesh directions only)
            // Z intermesh: 4 sender channels (3 mesh directions + Z)
            uint32_t mesh_vc1_sender_count = builder_config::num_downstream_edms_2d_vc1;  // default 3

            if (intermesh_vc_config_->router_type == IntermeshRouterType::Z_INTERMESH) {
                mesh_vc1_sender_count = 4;  // 3 mesh directions + Z
            }

            constexpr uint32_t mesh_vc1_base_sender_channel = builder_config::num_sender_channels_2d_mesh;
            constexpr uint32_t mesh_vc1_receiver_channel = 1;

            // TODO: Should these be VC relative? If so, we don't need to add the base sender channel.
            // Create sender channels (3 or 4 depending on router type)
            for (uint32_t i = 0; i < mesh_vc1_sender_count; ++i) {
                sender_channel_map_[LogicalSenderChannelKey{1, i}] =
                    InternalSenderChannelMapping{BuilderType::ERISC, mesh_vc1_base_sender_channel + i};
            }

            // Create ONE receiver channel for VC1 (not N)
            // A receiver channel forwards to multiple downstream sender channels
            receiver_channel_map_[LogicalReceiverChannelKey{1, 0}] =
                InternalReceiverChannelMapping{BuilderType::ERISC, mesh_vc1_receiver_channel};
        }
        // If intermesh VC not required or is inter-mesh router, don't create VC1 mappings
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

uint32_t FabricRouterChannelMapping::get_num_virtual_channels() const {
    // Z routers always have 2 VCs: VC0 (mesh traffic) and VC1 (Z traffic)
    if (is_z_router()) {
        return 2;
    }

    // Check if intermesh VC is required
    if (intermesh_vc_config_ && intermesh_vc_config_->requires_vc1) {
        // Pass-through mode: all routers have VC0 and VC1
        if (intermesh_vc_config_->requires_vc1_mesh_pass_through) {
            log_trace(tt::LogFabric, "get_num_virtual_channels: pass_through mode → 2 VCs");
            return 2;
        }

        // Other modes (EDGE_ONLY, FULL_MESH): Inter-mesh only has VC0, intra-mesh has VC0+VC1
        // Use the stored value (set during construction) to match what was actually created
        if (is_inter_mesh_router_) {
            log_trace(tt::LogFabric, "get_num_virtual_channels: is_inter_mesh_router_=true → 1 VC");
            return 1;  // Inter-mesh routers only have VC0
        } else {
            log_trace(tt::LogFabric, "get_num_virtual_channels: is_inter_mesh_router_=false → 2 VCs");
            return 2;  // Intra-mesh routers have VC0 + VC1
        }
    }

    log_trace(tt::LogFabric, "get_num_virtual_channels: no intermesh VC → 1 VC");
    return 1;  // VC0 only (single-mesh or 1D)
}

uint32_t FabricRouterChannelMapping::get_num_sender_channels_for_vc(uint32_t vc) const {
    constexpr uint32_t vc1_index = 1;
    constexpr uint32_t no_channels = 0;

    switch (vc) {
        case 0:  // VC0
            return builder_config::get_num_used_sender_channel_count(get_topology());
        case 1:  // VC1
            if (is_z_router()) {
                return builder_config::num_sender_channels_z_router_vc1;
            } else if (is_2D_topology(topology_)) {
                // Check if VC1 mappings were actually created in initialize_vc1_mappings()
                // Count how many sender channels exist for VC1
                LogicalSenderChannelKey test_key{vc1_index, 0};
                if (sender_channel_map_.find(test_key) == sender_channel_map_.end()) {
                    return no_channels;  // VC1 not enabled (no mappings created)
                }

                // Count actual sender channels (3 for XY intermesh, 4 for Z intermesh)
                uint32_t count = 0;
                for (uint32_t i = 0; i < builder_config::num_downstream_edms_2d_vc1_with_z; ++i) {
                    if (sender_channel_map_.find(LogicalSenderChannelKey{vc1_index, i}) != sender_channel_map_.end()) {
                        count++;
                    } else {
                        break;  // Channels are created sequentially, so stop at first missing
                    }
                }
                return count;
            }
            return no_channels;  // 1D topologies don't have VC1
        default:
            return no_channels;
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
