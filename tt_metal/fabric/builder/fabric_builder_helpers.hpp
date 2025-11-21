// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/fabric_common.h"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/fabric_context.hpp"

namespace tt::tt_fabric {

inline uint32_t get_worker_connected_sender_channel() {
    // Sender channel 0 is always for local worker in the new design
    return 0;
}

inline uint32_t get_vc1_connected_sender_channel(Topology topology) {
    if (topology == tt::tt_fabric::Topology::Ring) {
        return builder_config::num_sender_channels_1d_ring - 1;  // channel 2 (last of 3)
    } else if (topology == tt::tt_fabric::Topology::Torus) {
        return builder_config::num_sender_channels_2d_torus - 1;  // channel 4 (last of 5)
    }
    return 0;  // invalid
}

inline uint32_t get_worker_or_vc1_connected_sender_channel(Topology topology) {
    uint32_t target_channel = get_worker_connected_sender_channel();
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

// This helper returns the sender channel on the downstream router that should receive traffic from the upstream router.
inline uint32_t get_downstream_edm_sender_channel(
    const bool is_2D_routing,
    const eth_chan_directions receiver_direction,
    const eth_chan_directions downstream_direction) {
    if (!is_2D_routing) {
        return 1;  // 1D: sender channel 1 for forwarding
    }

    // Sender channel 0 is always reserved for the local worker.
    //
    // Sender channels 1–3 correspond to the three upstream neighbors relative
    // to the downstream router’s direction.
    //
    // The mapping from receiver direction → sender channel depends on the
    // downstream forwarding direction:
    //
    //   • Downstream = EAST:
    //         WEST  → channel 1
    //         NORTH → channel 2
    //         SOUTH → channel 3
    //
    //   • Downstream = WEST:
    //         EAST  → channel 1
    //         NORTH → channel 2
    //         SOUTH → channel 3
    //
    //   • Downstream = NORTH:
    //         EAST  → channel 1
    //         WEST  → channel 2
    //         SOUTH → channel 3
    //
    //   • Downstream = SOUTH:
    //         EAST  → channel 1
    //         WEST  → channel 2
    //         NORTH → channel 3

    size_t downstream_compact_index_for_upstream;
    if (downstream_direction == eth_chan_directions::EAST) {
        // EAST downstream: WEST(1)→0, NORTH(2)→1, SOUTH(3)→2
        downstream_compact_index_for_upstream = receiver_direction - 1;
    } else {
        // For other downstream directions: if upstream < downstream, use as-is; else subtract 1
        downstream_compact_index_for_upstream =
            (receiver_direction < downstream_direction) ? receiver_direction : (receiver_direction - 1);
    }

    // Sender channel = 1 + compact index (since channel 0 is for local worker)
    return 1 + downstream_compact_index_for_upstream;
}

// A receiver channel has 3 downstream EDMs
// This helper returns the index at which the receiver channel should store the downstream EDMs information.
// The index is 0, 1, 2 and depends on the downstream direction relative to the my direction.
inline size_t get_receiver_channel_compact_index(
    const eth_chan_directions receiver_direction, const eth_chan_directions downstream_direction) {
    size_t compact_index;
    if (receiver_direction == 0) {
        compact_index = downstream_direction - 1;
    } else {
        compact_index = (downstream_direction < receiver_direction) ? downstream_direction : (downstream_direction - 1);
    }
    return compact_index;
}
}  // namespace tt::tt_fabric
