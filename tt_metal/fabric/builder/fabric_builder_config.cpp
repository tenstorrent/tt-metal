// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_builder_config.hpp"
#include "fabric_tensix_builder_impl.hpp"
#include "tt_metal/fabric/fabric_context.hpp"

#include <cstdint>

namespace tt::tt_fabric::builder_config {

uint32_t get_sender_channel_count(const bool is_2D_routing) {
    return is_2D_routing ? builder_config::num_sender_channels_2d : builder_config::num_sender_channels_1d;
}

uint32_t get_receiver_channel_count(const bool is_2D_routing) {
    return is_2D_routing ? builder_config::num_receiver_channels_2d : builder_config::num_receiver_channels_1d;
}

uint32_t get_num_used_sender_channel_count(const Topology topology) {
    switch (topology) {
        case Topology::NeighborExchange: return builder_config::num_sender_channels_1d_neighbor_exchange;
        case Topology::Linear:
        case Topology::Ring: return builder_config::num_sender_channels_1d_linear;
        case Topology::Mesh:
        case Topology::Torus: return builder_config::num_sender_channels_2d_mesh;
        default: TT_THROW("unknown fabric topology: {}", topology); break;
    }
}

uint32_t get_num_tensix_sender_channels(Topology topology, tt::tt_fabric::FabricTensixConfig fabric_tensix_config) {
    // TODO: once we support inserting tensix as downstream in UDM mode, add back the channel count for UDM mode
    TT_FATAL(
        fabric_tensix_config == tt::tt_fabric::FabricTensixConfig::MUX,
        "get_num_tensix_sender_channels only supports MUX mode, got {}",
        static_cast<uint32_t>(fabric_tensix_config));

    // MUX mode: use topology-based channel count
    return get_num_used_sender_channel_count(topology);
}

uint32_t get_downstream_edm_count(bool is_2D_routing) {
    return is_2D_routing ? builder_config::num_downstream_edms_2d : builder_config::num_downstream_edms_1d;
}

uint32_t get_vc0_downstream_edm_count(bool is_2D_routing) {
    return is_2D_routing ? builder_config::num_downstream_edms_2d_vc0 : builder_config::num_downstream_edms_vc0;
}

uint32_t get_vc1_downstream_edm_count(bool is_2D_routing) {
    TT_FATAL(is_2D_routing, "VC1 is only supported for 2D routing");
    return builder_config::num_downstream_edms_2d_vc1;
}

}  // namespace tt::tt_fabric::builder_config
