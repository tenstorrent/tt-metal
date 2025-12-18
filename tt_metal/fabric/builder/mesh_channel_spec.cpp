// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_channel_spec.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"

namespace tt::tt_fabric {

MeshChannelSpec MeshChannelSpec::create_for_compute_mesh(Topology topology, const IntermeshVCConfig* intermesh_config) {
    MeshChannelSpec spec;
    const bool is_2d = is_2D_topology(topology);

    // Standard mesh router configuration
    // VC0: always present
    switch (topology) {
        case Topology::NeighborExchange:
            spec.sender_channels_per_vc[0] = 1;
            spec.downstream_edms_per_vc[0] = 0;  // No forwarding
            break;
        case Topology::Linear:
        case Topology::Ring:
            spec.sender_channels_per_vc[0] = 2;  // worker + forwarding
            spec.downstream_edms_per_vc[0] = 1;  // 1 downstream
            break;
        case Topology::Mesh:
        case Topology::Torus:
            spec.sender_channels_per_vc[0] = 4;  // worker + 3 forwarding
            spec.downstream_edms_per_vc[0] = 3;  // 3 downstream (N/E/S or E/S/W etc.)
            break;
    }
    spec.receiver_channels_per_vc[0] = 1;
    spec.num_vcs = 1;

    // Z router VC0 configuration
    if (is_2d) {
        spec.z_router_sender_channels_per_vc[0] = builder_config::num_sender_channels_z_router_vc0;
        spec.z_router_receiver_channels_per_vc[0] = 1;
    }

    // VC1: only for 2D topologies with intermesh
    if (is_2d && intermesh_config && intermesh_config->requires_vc1) {
        spec.num_vcs = 2;

        // Standard mesh router VC1
        if (intermesh_config->router_type == IntermeshRouterType::Z_INTERMESH) {
            spec.sender_channels_per_vc[1] = 4;  // 3 mesh + Z
            spec.downstream_edms_per_vc[1] = 4;  // 3 mesh directions + Z
        } else {
            spec.sender_channels_per_vc[1] = 3;  // 3 mesh directions
            spec.downstream_edms_per_vc[1] = 3;  // 3 mesh directions
        }
        spec.receiver_channels_per_vc[1] = 1;

        // Z router VC1 configuration
        spec.z_router_sender_channels_per_vc[1] = builder_config::num_sender_channels_z_router_vc1;
        spec.z_router_receiver_channels_per_vc[1] = 1;
    }

    spec.validate();
    return spec;
}

}  // namespace tt::tt_fabric
