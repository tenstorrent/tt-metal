// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// A fabric packet built host side for injection into a router's receiver channel; the router cannot
// tell it from a cabled arrival. Only a synthesising caller needs it -- a forwarded frame has one.
#pragma once

#include <cstdint>
#include <cstring>

#include "llrt/hal.hpp"
#include "fabric/fabric_edm_packet_header.hpp"
#include "hostdevcommon/fabric_common.h"

namespace tt::tt_fabric::erisc_bridge {

using LL = tt::tt_fabric::RoutingFieldsConstants::LowLatency;

// 36, NOT 32: noc_xy_encoding() returns the raw (y << NODE_ID_BITS) | x and does not pre-shift.
// At 32 the coordinates land in the local-address field and nothing arrives, silently. §7.1d.
inline constexpr std::uint32_t kNocAddrLocalBits = 36;

inline std::uint64_t noc_addr(
    const tt::tt_metal::Hal& hal,
    std::uint32_t noc_x,
    std::uint32_t noc_y,
    std::uint32_t l1_addr,
    std::uint32_t local_bits = kNocAddrLocalBits) {
    return (static_cast<std::uint64_t>(hal.noc_xy_encoding(noc_x, noc_y)) << local_bits) |
           static_cast<std::uint64_t>(l1_addr);
}

// Fills `storage` as to_noc_unicast_write() plus a terminal route would. Takes RAW storage
// because LowLatencyPacketHeaderT has a deleted default ctor -- write it where it will live.
template <typename HeaderT>
inline bool make_local_write_header(
    void* storage,
    const tt::tt_metal::Hal& hal,
    std::uint32_t dst_noc_x,
    std::uint32_t dst_noc_y,
    std::uint32_t dst_l1_addr,
    std::uint32_t payload_bytes) {
    if (payload_bytes == 0 || payload_bytes > 0xFFFFu) {
        return false;  // uint16_t field: a larger payload wraps and the router truncates silently
    }
    // memset through void*: the typed form trips -Wclass-memaccess on a non-trivial header.
    std::memset(storage, 0, sizeof(HeaderT));
    HeaderT& hdr = *static_cast<HeaderT*>(storage);
    hdr.noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;  // unscoped enum
    hdr.command_fields.unicast_write.noc_address = noc_addr(hal, dst_noc_x, dst_noc_y, dst_l1_addr);
    hdr.payload_size_bytes = static_cast<std::uint16_t>(payload_bytes);
    // WRITE_ONLY, not WRITE_AND_FORWARD: this packet terminates here. A forward bit would send
    // it down a link with no peer expecting it.
    hdr.routing_fields.value = LL::WRITE_ONLY;
    return true;
}

}  // namespace tt::tt_fabric::erisc_bridge
