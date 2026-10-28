// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Routing: what a receiver does with a packet, decided from its header. The answer is a Route: deliver it to this
// chip or not, and which downstreams to forward it to (one bit per downstream, named as router_downstream.hpp names
// them: by direction in 2D, 0 in 1D). 1D and 2D differ only here.
//
// A Route holds the packet's routing value as read (2D: the action byte; 1D: the routing field) and answers each
// question with a mask of it, so a test costs what testing the header value would. (A Route that packed the answers
// into one word cost about five instructions per packet to build and unpack.)
//   - 2D: the packet's route buffer holds one action byte per mesh row (Y) and column (X). An E/W-facing router reads
//     its column's byte; any other router reads its row's byte, and its column's when the row's is 0 (Y first). The
//     action has one bit per output direction (bit d for direction d) and a local-delivery bit, so its direction bits
//     are the downstreams to forward to.
//   - 1D: the routing field says write here, forward, or both.

#include "router_common.hpp"
#include "router_downstream.hpp"

namespace fabric_router {

#if defined(FABRIC_2D)
static_assert(Routing2DCodec::action_bit(eth_chan_directions::EAST) == 1u << eth_chan_directions::EAST);
static_assert(Routing2DCodec::action_bit(eth_chan_directions::Z) == 1u << eth_chan_directions::Z);

template <size_t VC>
struct Routing2D {
    // The action byte. E/W-facing routers decide on the column's byte alone. The others read both bytes as volatile
    // loads, unconditionally, so they issue together; a destination reached on the Y leg decides on the X byte, which
    // would otherwise wait behind the Y byte's test (LEDGER E1-19).
    static FORCE_INLINE uint8_t action(tt_l1_ptr PACKET_HEADER_TYPE* header, uint8_t my_y, uint8_t my_x) {
        if constexpr (my_direction == eth_chan_directions::EAST || my_direction == eth_chan_directions::WEST) {
            return header->route_buffer[MESH_Y_SIZE + my_x];
        } else {
            const volatile PACKET_HEADER_TYPE* const h = header;
            const uint8_t y_action = h->route_buffer[my_y];
            const uint8_t x_action = h->route_buffer[MESH_Y_SIZE + my_x];
            return y_action != 0 ? y_action : x_action;
        }
    }

    // The packet's routing fields as this router uses them: its action byte, read once per packet.
    using Fields = uint32_t;
    static FORCE_INLINE Fields read(tt_l1_ptr PACKET_HEADER_TYPE* header, uint8_t my_y, uint8_t my_x) {
        return action(header, my_y, my_x);
    }

    // A packet's route: its action byte.
    class Route {
    public:
        FORCE_INLINE explicit Route(uint32_t action) : action_(action) {}
        // The downstreams to forward to: the action's direction bits, live ones only (never this router's own).
        FORCE_INLINE uint32_t forward_to() const { return action_ & Downstreams<VC>::live; }
        FORCE_INLINE bool forwards() const { return forward_to() != 0; }
        FORCE_INLINE bool delivers() const { return (action_ & Routing2DCodec::ACTION_LOCAL_DELIVER) != 0; }

    private:
        uint32_t action_;
    };

    static FORCE_INLINE Route decide(Fields a) { return Route(a); }

    // Nothing in a 2D header changes on the way.
    static FORCE_INLINE void before_forward(tt_l1_ptr PACKET_HEADER_TYPE*, Fields) {}
};
#else
using LowLatencyFields = tt::tt_fabric::RoutingFieldsConstants::LowLatency;
static_assert(LowLatencyFields::WRITE_AND_FORWARD == (LowLatencyFields::WRITE_ONLY | LowLatencyFields::FORWARD_ONLY));

struct Routing1D {
    // The packet's routing fields, read once per packet: the decision and the next hop's fields both come from them.
    using Fields = ROUTING_FIELDS_TYPE;
    static FORCE_INLINE Fields read(tt_l1_ptr PACKET_HEADER_TYPE* header, uint8_t, uint8_t) {
        return header->routing_fields;
    }

    // A packet's route: this hop's routing field.
    class Route {
    public:
        FORCE_INLINE explicit Route(uint32_t field) : field_(field) {}
        // Downstream 0, the one downstream, when the packet goes on.
        FORCE_INLINE uint32_t forward_to() const { return forwards() ? 0x1u : 0u; }
        FORCE_INLINE bool forwards() const { return (field_ & LowLatencyFields::FORWARD_ONLY) != 0; }
        FORCE_INLINE bool delivers() const { return (field_ & LowLatencyFields::WRITE_ONLY) != 0; }

    private:
        uint32_t field_;
    };

    static FORCE_INLINE Route decide(const Fields& fields) {
        const uint32_t field = fields.value & LowLatencyFields::FIELD_MASK;
        ASSERT(field != LowLatencyFields::NOOP);
        return Route(field);
    }

    // The next router reads the next hop's field: this hop's is consumed, from the fields read with the decision.
    static FORCE_INLINE void before_forward(tt_l1_ptr PACKET_HEADER_TYPE* header, const Fields& fields) {
        update_packet_header_for_next_hop(header, fields);
    }
};
#endif

}  // namespace fabric_router
