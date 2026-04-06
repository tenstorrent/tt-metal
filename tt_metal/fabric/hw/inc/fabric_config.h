// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/fabric_common.h"
#include "dev_mem_map.h"

namespace tt::tt_fabric {

/**
 * Get maximum fabric packet payload size from runtime configuration.
 * Reads from the first valid fabric connection in L1 memory.
 * The buffer_size_bytes includes header, so we subtract sizeof(PACKET_HEADER_TYPE).
 *
 * @return Maximum packet payload size in bytes
 */
FORCE_INLINE uint32_t get_fabric_max_packet_size() {
    tt_l1_ptr tensix_fabric_connections_l1_info_t* connection_info =
        reinterpret_cast<tt_l1_ptr tensix_fabric_connections_l1_info_t*>(MEM_TENSIX_FABRIC_CONNECTIONS_BASE);

    uint32_t valid_mask = connection_info->valid_connections_mask;

    // Find first valid connection
    for (uint8_t i = 0; i < tensix_fabric_connections_l1_info_t::MAX_FABRIC_ENDPOINTS; i++) {
        if (valid_mask & (1 << i)) {
            uint32_t buffer_size = connection_info->read_only[i].buffer_size_bytes;
            uint32_t header_size = sizeof(PACKET_HEADER_TYPE);
            uint32_t max_packet_size = buffer_size - header_size;

            return max_packet_size;
        }
    }

    // No valid connections found
    ASSERT(false && "No valid fabric connections found in configuration");
    return 0;  // Unreachable, but satisfies compiler
}

}  // namespace tt::tt_fabric
