// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric/fabric_edm_packet_header.hpp"
#include "api/debug/assert.h"
#include "api/debug/dprint.h"

namespace tt::tt_fabric {

// TX-side checks that reject a header corrupted before injection, so it never hangs a downstream
// router. On failure DPRINT names the field, then return false so the caller's ASSERT trips.

// Verifies that the header's payload_size_bytes is consistent with the actual transfer size.
template <typename HeaderT>
FORCE_INLINE bool is_valid_payload_size(const HeaderT& packet_header, size_t transfer_size_bytes) {
    if (packet_header.get_payload_size_including_header() != transfer_size_bytes) {
        DPRINT(
            "FABRIC TX INVALID HDR: payload_size_bytes={} inconsistent with transfer\n",
            (uint32_t)packet_header.payload_size_bytes);
        return false;
    }
    return true;
}

// Fields shared by all header types (from PacketHeaderBase).
template <typename HeaderT>
FORCE_INLINE bool check_if_valid(const HeaderT& packet_header) {
    if (packet_header.noc_send_type > NOC_SEND_TYPE_LAST) {
        DPRINT("FABRIC TX INVALID HDR: bad noc_send_type={}\n", (uint32_t)packet_header.noc_send_type);
        return false;
    }
    if ((packet_header.noc_send_type == NOC_UNICAST_ATOMIC_INC ||
         packet_header.noc_send_type == NOC_UNICAST_INLINE_WRITE) &&
        packet_header.payload_size_bytes != 0) {
        DPRINT(
            "FABRIC TX INVALID HDR: nonzero payload_size_bytes={} for header-only op\n",
            (uint32_t)packet_header.payload_size_bytes);
        return false;
    }
    return true;
}

FORCE_INLINE bool is_valid(const PacketHeader& packet_header) {
    if (packet_header.chip_send_type > CHIP_SEND_TYPE_LAST) {
        DPRINT("FABRIC TX INVALID HDR: bad chip_send_type={}\n", (uint32_t)packet_header.chip_send_type);
        return false;
    }
    return check_if_valid(packet_header);
}
FORCE_INLINE void validate(const PacketHeader& packet_header) { ASSERT(is_valid(packet_header)); }

FORCE_INLINE bool is_valid(const LowLatencyPacketHeader& packet_header) { return check_if_valid(packet_header); }
FORCE_INLINE void validate(const LowLatencyPacketHeader& packet_header) { ASSERT(is_valid(packet_header)); }

FORCE_INLINE bool is_valid(const HybridMeshPacketHeader& packet_header) {
    // OOB hop_index indexes route_buffer OOB on RX -> decodes NOOP -> head-of-line livelock.
    if (packet_header.routing_fields.hop_index >= get_max_num_hops<HybridMeshPacketHeader>::value) {
        DPRINT("FABRIC TX INVALID HDR: hop_index={} out of bounds\n", (uint32_t)packet_header.routing_fields.hop_index);
        return false;
    }
    return check_if_valid(packet_header);
}
FORCE_INLINE void validate(const HybridMeshPacketHeader& packet_header) { ASSERT(is_valid(packet_header)); }

}  // namespace tt::tt_fabric
