// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/fabric_context.hpp"

namespace tt::tt_fabric {

inline uint32_t get_worker_connected_sender_channel(const eth_chan_directions direction, Topology topology) {
    const bool is_2D_routing = FabricContext::is_2D_topology(topology);
    return is_2D_routing ? direction : 0;
}

inline uint32_t get_vc1_connected_sender_channel(Topology topology) {
    if (topology == tt::tt_fabric::Topology::Ring) {
        return builder_config::num_sender_channels_1d_ring - 1;  // channel 2 (last of 3)
    } else if (topology == tt::tt_fabric::Topology::Torus) {
        return builder_config::num_sender_channels_2d_torus - 1;  // channel 4 (last of 5)
    }
    return 0;  // invalid
}

inline uint32_t get_worker_or_vc1_connected_sender_channel(const eth_chan_directions direction, Topology topology) {
    uint32_t target_channel = get_worker_connected_sender_channel(direction, topology);
    // if without vc1, return worker channel, otherwise return vc1 channel
    if (topology == tt::tt_fabric::Topology::Ring) {
        return builder_config::num_sender_channels_1d_ring - 1;  // channel 2 (last of 3)
    } else if (topology == tt::tt_fabric::Topology::Torus) {
        return builder_config::num_sender_channels_2d_torus - 1;  // channel 4 (last of 5)
    }
    return target_channel;  // Default to target_channel for Linear/Mesh
}

inline size_t get_dateline_sender_channel_skip_idx(const bool is_2D_routing) {
    // Dateline channel skip indices
    static constexpr size_t dateline_sender_channel_skip_idx = 2;
    static constexpr size_t dateline_sender_channel_skip_idx_2d = 4;
    return is_2D_routing ? dateline_sender_channel_skip_idx_2d : dateline_sender_channel_skip_idx;
}

inline uint32_t get_downstream_edm_sender_channel(const bool is_2D_routing, const eth_chan_directions direction) {
    return is_2D_routing ? direction : 1;
}

}  // namespace tt::tt_fabric
