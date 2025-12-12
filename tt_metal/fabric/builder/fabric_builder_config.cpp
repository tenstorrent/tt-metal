// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_builder_config.hpp"
#include "fabric_tensix_builder_impl.hpp"

#include <cstdint>

namespace tt::tt_fabric::builder_config {

uint32_t get_sender_channel_count(const bool is_2D_routing) {
    return is_2D_routing ? builder_config::num_sender_channels_2d : builder_config::num_sender_channels_1d;
}

uint32_t get_receiver_channel_count(const bool is_2D_routing) {
    return is_2D_routing ? builder_config::num_receiver_channels_2d : builder_config::num_receiver_channels_1d;
}

std::array<uint32_t, 2> get_sender_channel_count_per_vc(const bool is_2D_routing) {
    if (is_2D_routing) {
        // 2D routing: VC0 has 4 sender channels (0-3), VC1 has 3 sender channels (4-6)
        // Channel 7 is reserved for future Z-axis routing
        // Total = 7 channels
        return {4, 3};
    } else {
        // 1D routing: VC0 has 2 sender channels, VC1 has 0
        return {builder_config::num_sender_channels_1d, 0};
    }
}

std::array<uint32_t, 2> get_receiver_channel_count_per_vc(const bool is_2D_routing) {
    if (is_2D_routing) {
        // 2D routing: VC0 has 1 receiver, VC1 has 1 receiver
        // Total = 2 receivers
        return {1, 1};
    } else {
        // 1D routing: VC0 has 1 receiver, VC1 has 0
        return {builder_config::num_receiver_channels_1d, 0};
    }
}

uint32_t get_num_tensix_sender_channels(Topology topology, tt::tt_fabric::FabricTensixConfig fabric_tensix_config) {
    // TODO: once we support inserting tensix as downstream in UDM mode, add back the channel count for UDM mode
    TT_FATAL(
        fabric_tensix_config == tt::tt_fabric::FabricTensixConfig::MUX,
        "get_num_tensix_sender_channels only supports MUX mode, got {}",
        static_cast<uint32_t>(fabric_tensix_config));

    uint32_t num_channels = 0;
    // MUX mode: use topology-based channel count
    switch (topology) {
        case tt::tt_fabric::Topology::Linear:
        case tt::tt_fabric::Topology::Ring:
            num_channels = tt::tt_fabric::builder_config::num_sender_channels_1d_linear;
            break;
        case tt::tt_fabric::Topology::Mesh:
        case tt::tt_fabric::Topology::Torus:
            num_channels = tt::tt_fabric::builder_config::num_sender_channels_2d_mesh;
            break;
        default: TT_THROW("unknown fabric topology: {}", topology); break;
    }
    return num_channels;
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
