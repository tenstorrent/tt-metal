// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric/fabric_edm_packet_header.hpp"
#include "api/debug/assert.h"

namespace tt::tt_fabric {

FORCE_INLINE void validate(const PacketHeader& packet_header) {
    ASSERT(packet_header.chip_send_type <= CHIP_SEND_TYPE_LAST);
}
FORCE_INLINE bool is_valid(const PacketHeader& packet_header) {
    // NOC_SPARSE_MCAST_WRITE is a 1D LowLatency-only type, so a non-LowLatency header is bounded by the
    // standard local-write range (NOC_UNICAST_SCATTER_WRITE) and must reject sparse.
    return (packet_header.chip_send_type <= CHIP_SEND_TYPE_LAST);
}

FORCE_INLINE void validate(const LowLatencyPacketHeader& packet_header) {}
FORCE_INLINE bool is_valid(const LowLatencyPacketHeader& packet_header) {
    // LowLatency is the only header allowed to carry NOC_SPARSE_MCAST_WRITE, which is NOC_SEND_TYPE_LAST.
    return (packet_header.noc_send_type <= NOC_SEND_TYPE_LAST);
}

FORCE_INLINE void validate(const HybridMeshPacketHeader& packet_header) {
    ASSERT(packet_header.noc_send_type <= NOC_SEND_TYPE_LAST);
}
FORCE_INLINE bool is_valid(const HybridMeshPacketHeader& packet_header) {
    return (packet_header.noc_send_type <= NOC_SEND_TYPE_LAST);
}

}  // namespace tt::tt_fabric
