// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/debug/assert.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "hostdevcommon/fabric_common.h"
#include "internal/risc_attribs.h"

namespace tt::tt_fabric {

namespace route_2d_detail {

FORCE_INLINE std::uint32_t widen_y_packed_byte(std::uint8_t packed) {
    return static_cast<std::uint32_t>(Routing2DCodec::widen_y(packed & 0x3u)) |
           (static_cast<std::uint32_t>(Routing2DCodec::widen_y((packed >> 2) & 0x3u)) << 8) |
           (static_cast<std::uint32_t>(Routing2DCodec::widen_y((packed >> 4) & 0x3u)) << 16) |
           (static_cast<std::uint32_t>(Routing2DCodec::widen_y((packed >> 6) & 0x3u)) << 24);
}

FORCE_INLINE std::uint32_t widen_x_packed_byte(std::uint8_t packed) {
    return static_cast<std::uint32_t>(Routing2DCodec::widen_x(packed & 0x3u)) |
           (static_cast<std::uint32_t>(Routing2DCodec::widen_x((packed >> 2) & 0x3u)) << 8) |
           (static_cast<std::uint32_t>(Routing2DCodec::widen_x((packed >> 4) & 0x3u)) << 16) |
           (static_cast<std::uint32_t>(Routing2DCodec::widen_x((packed >> 6) & 0x3u)) << 24);
}

// Streams widened action bytes into the packet's contiguous [Y | X] map. Full words start at the
// naturally aligned route_buffer base; the final 1-3 bytes use narrow stores so they cannot overwrite
// dst_start_node_id on the 67-byte header tier.
class Route2DWordWriter {
public:
    explicit FORCE_INLINE Route2DWordWriter(volatile tt_l1_ptr std::uint8_t* output) :
        output_words(reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(output)) {}

    FORCE_INLINE void append(std::uint32_t widened, std::uint32_t num_bytes) {
        if (num_bytes == 4 && pending_bytes == 0) {
            *output_words++ = widened;
            return;
        }

        if (num_bytes < 4) {
            widened &= (1u << (num_bytes * 8)) - 1u;
        }

        pending_word |= widened << (pending_bytes * 8);
        const std::uint32_t total_bytes = pending_bytes + num_bytes;
        if (total_bytes < 4) {
            pending_bytes = total_bytes;
            return;
        }

        *output_words++ = pending_word;
        const std::uint32_t consumed_bytes = 4 - pending_bytes;
        pending_word = num_bytes > consumed_bytes ? widened >> (consumed_bytes * 8) : 0;
        pending_bytes = total_bytes - 4;
    }

    FORCE_INLINE void flush() {
        volatile tt_l1_ptr std::uint8_t* output = reinterpret_cast<volatile tt_l1_ptr std::uint8_t*>(output_words);
        std::uint32_t value = pending_word;
        if (pending_bytes >= 2) {
            *reinterpret_cast<volatile tt_l1_ptr std::uint16_t*>(output) = static_cast<std::uint16_t>(value);
            output += 2;
            value >>= 16;
        }
        if (pending_bytes & 1u) {
            *output = static_cast<std::uint8_t>(value);
        }
    }

private:
    volatile tt_l1_ptr std::uint32_t* output_words;
    std::uint32_t pending_word = 0;
    std::uint32_t pending_bytes = 0;
};

}  // namespace route_2d_detail

// Installs destination dst_dev_id's action maps from the given route table: the Y row widened into
// route_buffer[0..Y), the X row into route_buffer[Y..Y+X), LOCAL_DELIVER OR-ed onto the destination's
// X slot. Shared by the worker unicast producer and the intermesh landing encoder.
inline void widen_2d_route_to_chip(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    const std::uint8_t* route_table,
    uint16_t dst_dev_id,
    uint8_t mesh_y_size,
    uint8_t mesh_x_size) {
    ASSERT(dst_dev_id < (uint32_t)mesh_y_size * mesh_x_size);
    ASSERT((uint32_t)mesh_y_size + mesh_x_size <= sizeof(packet_header->route_buffer));

    const uint32_t dst_y = dst_dev_id / mesh_x_size;
    const uint32_t dst_x = dst_dev_id % mesh_x_size;

    route_2d_detail::Route2DWordWriter output(packet_header->route_buffer);

    const std::uint8_t* y_vec = Routing2DCodec::y_row(route_table, mesh_y_size, dst_y);
    const std::uint32_t y_full_bytes = mesh_y_size / Routing2DCodec::ACTIONS_PER_BYTE;
    for (std::uint32_t i = 0; i < y_full_bytes; ++i) {
        output.append(route_2d_detail::widen_y_packed_byte(y_vec[i]), Routing2DCodec::ACTIONS_PER_BYTE);
    }
    const std::uint32_t y_tail = mesh_y_size % Routing2DCodec::ACTIONS_PER_BYTE;
    if (y_tail != 0) {
        output.append(route_2d_detail::widen_y_packed_byte(y_vec[y_full_bytes]), y_tail);
    }

    const std::uint8_t* x_vec = Routing2DCodec::x_row(route_table, mesh_y_size, mesh_x_size, dst_x);
    const std::uint32_t dst_x_byte = dst_x / Routing2DCodec::ACTIONS_PER_BYTE;
    const std::uint32_t dst_x_shift = (dst_x % Routing2DCodec::ACTIONS_PER_BYTE) * 8;
    const std::uint32_t x_full_bytes = mesh_x_size / Routing2DCodec::ACTIONS_PER_BYTE;
    for (std::uint32_t i = 0; i < x_full_bytes; ++i) {
        std::uint32_t widened = route_2d_detail::widen_x_packed_byte(x_vec[i]);
        if (i == dst_x_byte) {
            widened |= static_cast<std::uint32_t>(Routing2DCodec::ACTION_LOCAL_DELIVER) << dst_x_shift;
        }
        output.append(widened, Routing2DCodec::ACTIONS_PER_BYTE);
    }
    const std::uint32_t x_tail = mesh_x_size % Routing2DCodec::ACTIONS_PER_BYTE;
    if (x_tail != 0) {
        std::uint32_t widened = route_2d_detail::widen_x_packed_byte(x_vec[x_full_bytes]);
        if (x_full_bytes == dst_x_byte) {
            widened |= static_cast<std::uint32_t>(Routing2DCodec::ACTION_LOCAL_DELIVER) << dst_x_shift;
        }
        output.append(widened, x_tail);
    }
    output.flush();
}

}  // namespace tt::tt_fabric
