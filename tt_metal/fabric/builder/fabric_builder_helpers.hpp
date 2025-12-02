// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/fabric_common.h"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/fabric_context.hpp"

namespace tt::tt_fabric {

namespace builder {

bool is_east_or_west(eth_chan_directions direction);
bool is_north_or_south(eth_chan_directions direction);
eth_chan_directions get_sender_channel_direction(eth_chan_directions my_direction, size_t sender_channel_index);
}  // namespace builder

inline uint32_t get_worker_connected_sender_channel() {
    // Sender channel 0 is always for local worker in the new design
    return 0;
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
