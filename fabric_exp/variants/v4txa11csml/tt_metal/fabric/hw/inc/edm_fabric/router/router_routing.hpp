// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Routing: what a receiver does with a packet, decided from its header. The answer is a Route: deliver it to this
// chip or not, and which downstreams to forward it to (one bit per downstream, named as router_downstream.hpp names
// them: by direction in 2D, 0 in 1D). 1D and 2D differ only here.
//
// `read` reads a packet's route word from its header, once per packet: one word, which the receiver may keep with the
// packet's header word until it handles the packet (router_receiver.hpp). A Route holds that word and answers each
// question with a mask of it, so a test costs what testing the header value would (packing the answers into a word of
// their own would cost instructions per packet to build and unpack).
//   - 2D: the packet's route buffer holds one action byte per mesh row (Y) and column (X). An E/W-facing router reads
//     its column's byte; any other router reads its row's byte, and its column's when the row's is 0 (Y first). The
//     action has one bit per output direction (bit d for direction d) and a local-delivery bit, so its direction bits
//     are the downstreams to forward to. The route word is the action byte.
//   - 1D: the routing word; its low field says write here, forward, or both (the next hops' fields are above it).
//
// Multi-mesh fabrics (2D) add two steps, both routing policy of the router kinds that need them, both inside `read`:
//   - landing (a receiver channel that is intermesh ingress: its packets come from another mesh): before the decode,
//     the packet's route buffer is rebuilt for this mesh (fabric_set_2d_intermesh_landing_route, out of line, every
//     packet on such a channel); its decode is then Y first whatever this router's direction;
//   - exit (a router on a chip at its mesh's edge): a packet whose action is "deliver here" while its destination mesh
//     is another one leaves the mesh here, forwarded as it is toward that mesh (the direction comes from the routing
//     table's intermesh directions, out of line), with no local delivery: its route word is the exit's direction bit.
//     The destination mesh is read with the action.

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp"
#if defined(FABRIC_2D)
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#endif

#include "router_common.hpp"
#include "router_downstream.hpp"

namespace fabric_router {

// This chip's place in the fabric, as routing uses it (2D): the routing table's my_mesh_coord_y / _x and my_mesh_id.
struct ChipCoordinates {
    uint8_t y;
    uint8_t x;
    uint16_t mesh_id;
};


#if defined(FABRIC_2D)
static_assert(Routing2DCodec::action_bit(eth_chan_directions::EAST) == 1u << eth_chan_directions::EAST);
static_assert(Routing2DCodec::action_bit(eth_chan_directions::Z) == 1u << eth_chan_directions::Z);

// The action of a packet leaving its mesh here toward mesh `dst_mesh`: the direction bit of that mesh's exit from the
// routing table. A direction this router cannot forward to (its own, or one without a downstream entry) means a bad
// routing table: an assert, and the packet is dropped (no forward, no delivery). Out of line: exits are rare.
template <size_t VC>
__attribute__((noinline)) uint32_t exit_action(uint32_t dst_mesh) {
    const auto& routing_table = l1_routing_table();
    const auto direction = static_cast<eth_chan_directions>(
        routing_table.inter_mesh_direction_table.get_original_direction(static_cast<uint16_t>(dst_mesh)));
    const bool addressable = direction < eth_chan_directions::COUNT &&
                             direction != static_cast<eth_chan_directions>(my_direction) &&
                             downstream_compact_index(direction) < DownstreamTraits<VC>::count;
    ASSERT(addressable);
    return addressable ? (1u << direction) : 0u;
}

// Routing on the downstreams of VC, for a receiver channel that is intermesh ingress (its packets come from another
// mesh; INTERMESH_INGRESS) or not.
template <size_t VC, bool INTERMESH_INGRESS>
struct Routing2D {
    static constexpr bool intermesh_exits = is_intramesh_router_on_edge;

    // The landing rewrite (out of line: one call per packet on an ingress channel).
    static __attribute__((noinline)) void set_landing_route(tt_l1_ptr PACKET_HEADER_TYPE* header) {
        fabric_set_2d_intermesh_landing_route(header, l1_routing_table(), MESH_Y_SIZE, MESH_X_SIZE);
    }

    // The action byte, by the codec's decode (Routing2DCodec, hostdevcommon/fabric_common.h; Y first after a landing).
    // Header bytes are written by the link's DMA, so the router reads them through the codec's volatile API; the codec
    // owns the decode.
    static FORCE_INLINE uint8_t action(tt_l1_ptr PACKET_HEADER_TYPE* header, const ChipCoordinates& my_chip) {
        const volatile uint8_t* const route_buffer = header->route_buffer;
        if constexpr (INTERMESH_INGRESS) {
            return Routing2DCodec::decode_action_y_first(route_buffer, my_chip.y, my_chip.x, MESH_Y_SIZE);
        } else {
            return Routing2DCodec::decode_action<static_cast<eth_chan_directions>(my_direction)>(
                route_buffer, my_chip.y, my_chip.x, MESH_Y_SIZE);
        }
    }

    // The packet's route word: its action byte, landed first on an ingress channel; on an edge router, the exit's
    // direction bit for a packet that leaves the mesh here (its destination mesh, a header field, read volatile as the
    // action bytes are).
    static FORCE_INLINE uint32_t read(tt_l1_ptr PACKET_HEADER_TYPE* header, const ChipCoordinates& my_chip) {
        if constexpr (INTERMESH_INGRESS) {
            set_landing_route(header);
        }
        if constexpr (intermesh_exits) {
            const volatile PACKET_HEADER_TYPE* const volatile_header = header;
            const uint32_t dst_mesh = volatile_header->dst_start_mesh_id;
            const uint32_t route_action = action(header, my_chip);
            return Routing2DCodec::action_is_intermesh_exit(route_action, dst_mesh, my_chip.mesh_id)
                       ? exit_action<VC>(dst_mesh)
                       : route_action;
        } else {
            return action(header, my_chip);
        }
    }

    // A packet's route: its route word.
    class Route {
    public:
        FORCE_INLINE explicit Route(uint32_t route_word) : action_(route_word) {}
        // The downstreams to forward to: the action's direction bits, live ones only (never this router's own).
        FORCE_INLINE uint32_t forward_to() const { return action_ & Downstreams<VC>::live; }
        FORCE_INLINE bool forwards() const { return forward_to() != 0; }
        FORCE_INLINE bool delivers() const { return (action_ & Routing2DCodec::ACTION_LOCAL_DELIVER) != 0; }

    private:
        uint32_t action_;
    };

    // Nothing in a 2D header changes on the way.
    static FORCE_INLINE void update_for_next_hop(tt_l1_ptr PACKET_HEADER_TYPE*, uint32_t) {}
};
#else
using LowLatencyFields = tt::tt_fabric::RoutingFieldsConstants::LowLatency;
static_assert(LowLatencyFields::WRITE_AND_FORWARD == (LowLatencyFields::WRITE_ONLY | LowLatencyFields::FORWARD_ONLY));

struct Routing1D {
    // The packet's route word: the routing word, whose low FIELD_WIDTH bits are this hop's field. One word, as 2D's
    // action: the extension words of packets past 16 hops are read only when the next-hop update needs them
    // (update_for_next_hop).
    static FORCE_INLINE uint32_t read(tt_l1_ptr PACKET_HEADER_TYPE* header, const ChipCoordinates&) {
        return header->routing_fields.value;
    }

    // A packet's route: this hop's field of its route word.
    class Route {
    public:
        FORCE_INLINE explicit Route(uint32_t route_word) : field_(route_word & LowLatencyFields::FIELD_MASK) {
            ASSERT(field_ != LowLatencyFields::NOOP);
        }
        // Downstream 0, the one downstream, when the packet goes on.
        FORCE_INLINE uint32_t forward_to() const { return forwards() ? 0x1u : 0u; }
        FORCE_INLINE bool forwards() const { return (field_ & LowLatencyFields::FORWARD_ONLY) != 0; }
        FORCE_INLINE bool delivers() const { return (field_ & LowLatencyFields::WRITE_ONLY) != 0; }

    private:
        uint32_t field_;
    };

    // The next router reads the next hop's field: this hop's is shifted out of the route word. When that empties the
    // word (a packet past 16 hops, with extension words), the first extension word refills it and the rest move down,
    // as update_packet_header_for_next_hop does; otherwise the extension words stay as they are in the header.
    // Volatile: the forward's NoC read of the slot must see the update.
    static FORCE_INLINE void update_for_next_hop(tt_l1_ptr PACKET_HEADER_TYPE* header, uint32_t route_word) {
        volatile tt_l1_ptr PACKET_HEADER_TYPE* const volatile_header = header;
        uint32_t next = route_word >> LowLatencyFields::FIELD_WIDTH;
#if defined(FABRIC_1D_PKT_HDR_EXTENSION_WORDS) && (FABRIC_1D_PKT_HDR_EXTENSION_WORDS > 0)
        constexpr uint32_t extension_words = FABRIC_1D_PKT_HDR_EXTENSION_WORDS;
        if (next == 0) {
            next = volatile_header->routing_fields.route_buffer[0];
            for (uint32_t i = 0; i + 1 < extension_words; i++) {
                volatile_header->routing_fields.route_buffer[i] = volatile_header->routing_fields.route_buffer[i + 1];
            }
            volatile_header->routing_fields.route_buffer[extension_words - 1] = 0;
        }
#endif
        volatile_header->routing_fields.value = next;
    }
};
#endif

}  // namespace fabric_router
