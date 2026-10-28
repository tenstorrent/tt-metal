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
//
// Multi-mesh fabrics (2D) add two steps, both routing policy of the router kinds that need them:
//   - landing (a receiver channel that is intermesh ingress: its packets come from another mesh): before the decode,
//     the packet's route buffer is rebuilt for this mesh (fabric_set_2d_intermesh_landing_route, out of line, every
//     packet on such a channel); its decode is then Y first whatever this router's direction;
//   - exit (a router on a chip at its mesh's edge): a packet whose action is "deliver here" while its destination mesh
//     is another one leaves the mesh here, forwarded as it is toward that mesh (the direction comes from the routing
//     table's intermesh directions, out of line), with no local delivery. The destination mesh is read with the action.

#include "router_common.hpp"
#include "router_downstream.hpp"

namespace fabric_router {

// This router's place in the fabric, as routing uses it (2D).
struct Coordinates {
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
    const auto& routing_table = *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const auto dir = static_cast<eth_chan_directions>(
        routing_table.inter_mesh_direction_table.get_original_direction(static_cast<uint16_t>(dst_mesh)));
    const bool addressable = dir < eth_chan_directions::COUNT && dir != static_cast<eth_chan_directions>(my_direction) &&
                             downstream_compact_index(dir) < DownstreamTraits<VC>::size;
    ASSERT(addressable);
    return addressable ? (1u << dir) : 0u;
}

// Routing on the downstreams of VC, for a receiver channel that is intermesh ingress (INGRESS) or not.
template <size_t VC, bool INGRESS>
struct Routing2D {
    // E/W-facing routers decide on the column's byte alone, except after a landing (Y first).
    static constexpr bool column_only =
        !INGRESS && (my_direction == eth_chan_directions::EAST || my_direction == eth_chan_directions::WEST);
    static constexpr bool exits = is_intramesh_router_on_edge;

    // The landing rewrite (out of line: one call per packet on an ingress channel).
    static __attribute__((noinline)) void land(tt_l1_ptr PACKET_HEADER_TYPE* header) {
        const auto& routing_table =
            *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
        fabric_set_2d_intermesh_landing_route(header, routing_table, MESH_Y_SIZE, MESH_X_SIZE);
    }

    // The action byte. Routers that read the row's byte read both bytes as volatile loads, unconditionally, so they issue
    // together; a destination reached on the Y leg decides on the X byte, which would otherwise wait behind the Y
    // byte's test (LEDGER E1-19).
    static FORCE_INLINE uint8_t action(tt_l1_ptr PACKET_HEADER_TYPE* header, const Coordinates& me) {
        if constexpr (column_only && !exits) {
            return header->route_buffer[MESH_Y_SIZE + me.x];
        } else if constexpr (column_only) {
            const volatile PACKET_HEADER_TYPE* const h = header;
            return h->route_buffer[MESH_Y_SIZE + me.x];
        } else {
            const volatile PACKET_HEADER_TYPE* const h = header;
            const uint8_t y_action = h->route_buffer[me.y];
            const uint8_t x_action = h->route_buffer[MESH_Y_SIZE + me.x];
            return y_action != 0 ? y_action : x_action;
        }
    }

    // The packet's routing fields as this router uses them, read once per packet: its action byte, and on an edge
    // router the destination mesh, read in the same group as the action bytes (both volatile) so the loads overlap.
    struct EdgeFields {
        uint32_t action;
        uint32_t dst_mesh;
    };
    using Fields = std::conditional_t<exits, EdgeFields, uint32_t>;
    static FORCE_INLINE Fields read(tt_l1_ptr PACKET_HEADER_TYPE* header, const Coordinates& me) {
        if constexpr (INGRESS) {
            land(header);
        }
        if constexpr (exits) {
            const volatile PACKET_HEADER_TYPE* const h = header;
            const uint32_t dst_mesh = h->dst_start_mesh_id;
            return {action(header, me), dst_mesh};
        } else {
            return action(header, me);
        }
    }

    // A packet's route: its action byte (an exit's: the exit direction's bit).
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

    static FORCE_INLINE Route decide(const Fields& f, const Coordinates& me) {
        if constexpr (exits) {
            if (Routing2DCodec::action_is_intermesh_exit(f.action, f.dst_mesh, me.mesh_id)) {
                return Route(exit_action<VC>(f.dst_mesh));
            }
            return Route(f.action);
        } else {
            return Route(f);
        }
    }

    // Nothing in a 2D header changes on the way.
    static FORCE_INLINE void before_forward(tt_l1_ptr PACKET_HEADER_TYPE*, const Fields&) {}
};
#else
using LowLatencyFields = tt::tt_fabric::RoutingFieldsConstants::LowLatency;
static_assert(LowLatencyFields::WRITE_AND_FORWARD == (LowLatencyFields::WRITE_ONLY | LowLatencyFields::FORWARD_ONLY));

struct Routing1D {
    // The packet's routing fields, read once per packet: the decision and the next hop's fields both come from them.
    using Fields = ROUTING_FIELDS_TYPE;
    static FORCE_INLINE Fields read(tt_l1_ptr PACKET_HEADER_TYPE* header, const Coordinates&) {
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

    static FORCE_INLINE Route decide(const Fields& fields, const Coordinates&) {
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
