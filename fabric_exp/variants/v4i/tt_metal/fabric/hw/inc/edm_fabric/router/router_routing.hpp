// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Routing: what a receiver does with a packet, decided from its header. The answer is a Route: deliver it to this
// chip or not, and which compact downstream slots to forward it to. 1D and 2D differ only here.
//   - 2D: the packet's route buffer holds one action byte per mesh row (Y) and column (X). An E/W-facing router reads
//     its column's byte; any other router reads its row's byte, and its column's when the row's is 0 (Y first). The
//     action has one bit per output direction (bit d for direction d) and a local-delivery bit.
//   - 1D: the routing field says write here, forward, or both.

#include "router_common.hpp"
#include "router_downstream.hpp"

namespace fabric_router {

// What to do with a packet, in one word: the compact downstream slots to forward to (live slots only, bits 0-3) and
// whether to write it to this chip (bit 31).
class Route {
public:
    static constexpr uint32_t DELIVER = 1u << 31;
    FORCE_INLINE constexpr Route(uint32_t bits) : bits_(bits) {}
    FORCE_INLINE constexpr uint32_t forward_slots() const { return bits_ & ~DELIVER; }
    FORCE_INLINE constexpr bool forwards() const { return forward_slots() != 0; }
    FORCE_INLINE constexpr bool delivers() const { return (bits_ & DELIVER) != 0; }

private:
    uint32_t bits_;
};

#if defined(FABRIC_2D)
// The compact downstream slots of a 2D action: direction d is slot d below this router's direction and d - 1 above it
// (downstream_compact_index), so the slots are the action's direction bits with this router's own bit removed.
constexpr uint32_t action_to_slots(uint32_t action) {
    constexpr uint32_t below = (1u << my_direction) - 1;
    const uint32_t eth = action & Routing2DCodec::ACTION_ETH_MASK;
    return (eth & below) | ((eth >> 1) & ~below);
}
static_assert(action_to_slots(Routing2DCodec::action_bit(downstream_direction(0))) == 0x1u);
static_assert(action_to_slots(Routing2DCodec::action_bit(downstream_direction(3))) == 0x8u);

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

    static FORCE_INLINE Route decide(Fields a) {
        static_assert(Routing2DCodec::ACTION_LOCAL_DELIVER == 1u << 5);
        return Route((action_to_slots(a) & Downstreams<VC>::live) | ((a & Routing2DCodec::ACTION_LOCAL_DELIVER) << 26));
    }

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

    static FORCE_INLINE Route decide(const Fields& fields) {
        const uint32_t field = fields.value & LowLatencyFields::FIELD_MASK;
        ASSERT(field != LowLatencyFields::NOOP);
        return Route(((field & LowLatencyFields::FORWARD_ONLY) != 0 ? 0x1u : 0u) |
                     ((field & LowLatencyFields::WRITE_ONLY) != 0 ? Route::DELIVER : 0u));
    }

    // The next router reads the next hop's field: this hop's is consumed, from the fields read with the decision.
    static FORCE_INLINE void before_forward(tt_l1_ptr PACKET_HEADER_TYPE* header, const Fields& fields) {
        update_packet_header_for_next_hop(header, fields);
    }
};
#endif

}  // namespace fabric_router
