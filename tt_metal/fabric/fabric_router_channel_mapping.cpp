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
    bool has_z_on_device) :
    topology_(topology),
    downstream_is_tensix_builder_(downstream_is_tensix_builder),
    variant_(variant),
    intermesh_vc_config_(intermesh_config),
    has_z_on_device_(has_z_on_device) {
    initialize_mappings();
}

void FabricRouterChannelMapping::initialize_mappings() {
    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        initialize_vc_mappings(vc);
    }
}

void FabricRouterChannelMapping::initialize_vc_mappings(uint32_t vc) {
    const bool is_2d = is_2D_topology(topology_);

    if (vc == 0) {
        // VC0 sender channel setup
        if (is_2d) {
            // 2D topology VC0 sender channels:
            //   Z_ROUTER: 5 channels (0=Worker, 1-4=E/W/N/S)
            //   MESH: 4 channels (0=Worker, 1-3=mesh directions)
            auto num_sender_channels = is_z_router()
                                           ? builder_config::num_sender_channels_z_router_per_vc[0]  // 5 channels
                                           : builder_config::num_sender_channels_2d_mesh;            // 4 channels

            log_debug(
                LogFabric,
                "initialize_vc_mappings: vc={}, variant={}, is_z_router={}, num_sender_channels={}",
                vc,
                static_cast<int>(variant_),
                is_z_router(),
                num_sender_channels);

            for (uint32_t i = 0; i < num_sender_channels; ++i) {
                // When mux extension is enabled, ALL VC0 channels go to TENSIX mux
                BuilderType builder_type = downstream_is_tensix_builder_ ? BuilderType::TENSIX : BuilderType::ERISC;
                sender_channel_map_[LogicalSenderChannelKey{vc, i}] = InternalSenderChannelMapping{builder_type, i};
            }
        } else if (topology_ == Topology::NeighborExchange) {
            // Neighbor Exchange topology VC0 has 1 sender channel:
            //  [0] = local worker channel
            TT_FATAL(!downstream_is_tensix_builder_, "Neighbor Exchange topology does not support mux extension");
            sender_channel_map_[LogicalSenderChannelKey{vc, 0}] = InternalSenderChannelMapping{BuilderType::ERISC, 0};
        } else {
            // 1D topology VC0 has 2 sender channels (relative indices within VC0):
            //   [0] = local worker channel
            //   [1] = forwarding channel from upstream router
            BuilderType vc0_builder_type = downstream_is_tensix_builder_ ? BuilderType::TENSIX : BuilderType::ERISC;
            sender_channel_map_[LogicalSenderChannelKey{vc, 0}] =
                InternalSenderChannelMapping{vc0_builder_type, 0};  // worker channel
            sender_channel_map_[LogicalSenderChannelKey{vc, 1}] =
                InternalSenderChannelMapping{vc0_builder_type, 1};  // forward channel
        }
        // Receiver channel (typically single receiver channel per VC)
        receiver_channel_map_[LogicalReceiverChannelKey{vc, 0}] = InternalReceiverChannelMapping{BuilderType::ERISC, 0};
    } else if (vc == 1) {
        // VC1+ (intermesh) only exists for 2D topologies and Z routers
        if (!is_2d) {
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
            // - Sender channels 0-3: Map to erisc internal channels 5-8 (mesh→Z traffic)
            // - Receiver channel 0: Maps to erisc internal channel 1 (Z→mesh traffic)
            uint32_t vc1_sender_count = builder_config::num_sender_channels_z_router_per_vc[vc];
            // Compute base sender channel as prefix sum of all previous VCs' sender counts
            uint32_t base_sender_channel = 0;
            for (uint32_t v = 0; v < vc; ++v) {
                base_sender_channel += builder_config::num_sender_channels_z_router_per_vc[v];
            }
            uint32_t receiver_channel = vc;  // VC0=0, VC1=1, etc.

            for (uint32_t i = 0; i < vc1_sender_count; ++i) {
                sender_channel_map_[LogicalSenderChannelKey{vc, i}] =
                    InternalSenderChannelMapping{BuilderType::ERISC, base_sender_channel + i};
            }

            receiver_channel_map_[LogicalReceiverChannelKey{vc, 0}] =
                InternalReceiverChannelMapping{BuilderType::ERISC, receiver_channel};
        } else {
            // Standard mesh router VC1+: create if intermesh VC is required
            if (intermesh_vc_config_ && intermesh_vc_config_->requires_vc1) {
                // Determine sender count based on whether device has Z router
                // Mesh with Z: 4 sender channels (3 mesh directions + 1 for Z router connection)
                // Mesh without Z: 3 sender channels (3 mesh directions only)
                uint32_t mesh_vc1_sender_count = has_z_on_device_ ? 4 : 3;

                uint32_t mesh_vc1_base_sender_channel = builder_config::num_sender_channels_2d_mesh;
                uint32_t mesh_vc1_receiver_channel = vc;  // VC0=0, VC1=1, etc.

                // Create sender channels (3 or 4 depending on router type)
                for (uint32_t i = 0; i < mesh_vc1_sender_count; ++i) {
                    sender_channel_map_[LogicalSenderChannelKey{vc, i}] =
                        InternalSenderChannelMapping{BuilderType::ERISC, mesh_vc1_base_sender_channel + i};
                }

                // Create ONE receiver channel for this VC
                receiver_channel_map_[LogicalReceiverChannelKey{vc, 0}] =
                    InternalReceiverChannelMapping{BuilderType::ERISC, mesh_vc1_receiver_channel};
            }
        }
    } else {
        TT_FATAL(false, "Invalid VC: {}. Currently unsupported", vc);
    }
}

bool FabricRouterChannelMapping::is_z_router() const {
    return variant_ == RouterVariant::Z_ROUTER;
}

bool FabricRouterChannelMapping::is_mesh_router() const { return variant_ == RouterVariant::MESH; }

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

    // Check if intermesh VC is required (all routers get VC1 when enabled)
    if (intermesh_vc_config_ && intermesh_vc_config_->requires_vc1) {
        return 2;  // Both inter-mesh and intra-mesh routers have VC0 + VC1
    }

    return 1;  // VC0 only (single-mesh or 1D)
}

uint32_t FabricRouterChannelMapping::get_num_sender_channels_for_vc(uint32_t vc) const {
    constexpr uint32_t vc1_index = 1;
    constexpr uint32_t no_channels = 0;

    switch (vc) {
        case 0:  // VC0
            if (is_z_router()) {
                // Only Z routers have 5 VC0 channels (0=Worker, 1-4=E/W/N/S)
                return builder_config::num_sender_channels_z_router_per_vc[0];  // 5 channels
            } else {
                // All mesh routers (MESH and MESH_AND_Z_ROUTER) have 4 VC0 channels
                return builder_config::get_num_used_sender_channel_count(get_topology());  // 4 channels
            }
        case 1:  // VC1
            if (is_z_router()) {
                return builder_config::num_sender_channels_z_router_per_vc[1];
            } else if (is_2D_topology(topology_)) {
                // Check if VC1 mappings were actually created in initialize_vc_mappings()
                // Count how many sender channels exist for VC1
                LogicalSenderChannelKey test_key{vc1_index, 0};
                if (!sender_channel_map_.contains(test_key)) {
                    return no_channels;  // VC1 not enabled (no mappings created)
                }

                // Count actual sender channels (3 for XY intermesh, 4 for Z intermesh)
                // 3 for MESH, 4 for MESH_AND_Z_ROUTER
                uint32_t count = 0;
                for (uint32_t i = 0; i < builder_config::num_downstream_edms_2d_vc1_with_z; ++i) {
                    if (sender_channel_map_.contains(LogicalSenderChannelKey{vc1_index, i})) {
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
