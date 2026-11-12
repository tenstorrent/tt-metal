// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A ring buffer slot's packet header as JSON: what the packet does where it delivers (its command) and where it's going
// (its route, as the header holds it). The header interpreters (fabric_decode_interpret.hpp) put these together into
// the slot's record. Each part reads the header the way the router kernel does, through the kernel's own types.

#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include <hostdevcommon/fabric_common.h>
#include <nlohmann/json.hpp>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"
#include "tt_metal/hw/inc/hostdev/fabric_edm_packet_header.hpp"

namespace tt::tt_fabric::decode {

using json = nlohmann::ordered_json;

struct InterpretContext;

// The members every header type shares, from PacketHeaderBase.
json make_packet_header_json(uint16_t payload_size_bytes, NocSendType noc_send_type, uint8_t src_ch_id);

// A NOC address split into the core's NOC coordinates and the address on that core, by the manifest's layout: the
// inverse of NOC_XY_ADDR. Written as { x, y, address }.
json make_noc_address_json(uint64_t noc_address, const manifest::NocAddressLayout& layout);

// What the packet does where it delivers: the NocCommandFields member the router reads for `noc_send_type`, with its
// NOC addresses split by context.arch.noc_address. Null for a send type with no member.
json make_noc_command_json(
    const NocCommandFields& command_fields, NocSendType noc_send_type, const InterpretContext& context);

// A 1D route's hop actions in the order the routers read them, up to and including the first that doesn't forward.
// Each router reads the low field of the routing word and shifts it out before forwarding; when that empties the word,
// the next extension word refills it (Routing1D::update_for_next_hop).
std::vector<RoutingFieldsConstants::LowLatency::HopAction> read_1d_hop_actions(
    uint32_t routing_word, std::span<const uint32_t> extension_words);

// A 1D route: { actions }, each hop action's name.
json make_1d_route_json(std::span<const RoutingFieldsConstants::LowLatency::HopAction> actions);

// The two maps of a 2D route buffer: the Y map, one action byte per row of `mesh`, then the X map, one per column, cut
// at the end of the buffer. Both are empty for a mesh that isn't 2D.
struct Route2DMaps {
    std::span<const uint8_t> y;
    std::span<const uint8_t> x;
};
Route2DMaps split_2d_route_maps(std::span<const uint8_t> route_buffer, const manifest::Mesh& mesh);

// A 2D action byte as names, lowest bit first: a direction's name for each Ethernet bit (east, west, north, south, z),
// "local" for local delivery, and "bit6", "bit7" for the reserved bits, so nothing in the byte is hidden.
json make_2d_action_json(uint8_t action);

// Every name make_2d_action_json writes, lowest bit first, for analysis.vocabulary.
json make_2d_action_vocabulary_json();

// A 2D route: { y, x, dst, mcast }. `y` and `x` are the maps' action bytes as names; `dst` is the destination chip's
// path (for a multicast, the anchor the map was built from); `mcast` is the hops in each direction, or null when every
// count is zero (a unicast). `mcast_params` is indexed by eth_chan_directions, as the header's is.
json make_2d_route_json(const Route2DMaps& maps, FabricNodeId dst_start, const std::array<uint16_t, 4>& mcast_params);

}  // namespace tt::tt_fabric::decode
