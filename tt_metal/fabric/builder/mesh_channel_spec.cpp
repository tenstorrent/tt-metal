// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_channel_spec.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"

namespace tt::tt_fabric {

MeshChannelSpec MeshChannelSpec::create_for_compute_mesh(Topology topology) {
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

    // VC1: ALWAYS populate for 2D topologies (max capacity)
    // Z routers always need VC1, so capacity must include it
    if (is_2d) {
        spec.num_vcs = 2;

        // Standard mesh router VC1 - use MAX capacity
        // Max is 4 channels (Z intermesh case: 3 mesh directions + Z)
        spec.sender_channels_per_vc[1] = 4;  // MAX: 3 mesh + Z
        spec.downstream_edms_per_vc[1] = 4;  // MAX: 3 mesh directions + Z
        spec.receiver_channels_per_vc[1] = 1;

        // Z router configuration (VC0 and VC1)
        spec.z_router_sender_channels_per_vc[0] = builder_config::num_sender_channels_z_router_vc0;
        spec.z_router_receiver_channels_per_vc[0] = 1;
        spec.z_router_sender_channels_per_vc[1] = builder_config::num_sender_channels_z_router_vc1;
        spec.z_router_receiver_channels_per_vc[1] = 1;
    }

    spec.validate();
    return spec;
}

}  // namespace tt::tt_fabric
