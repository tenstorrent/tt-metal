// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "api/debug/assert.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "hostdevcommon/fabric_common.h"
#include "internal/risc_attribs.h"

namespace tt::tt_fabric {

namespace route_2d_detail {

// Device route tables place both reverse-tree regions on at least a halfword boundary. memcpy keeps
// the raw byte storage alias-safe and gives the device compiler an opportunity to fold the copy into
// one little-endian halfword load when it proves that alignment through the inlined call chain.
struct AlignedMcastTreeEdgeReader {
    static FORCE_INLINE std::uint16_t get(const std::uint8_t* region, std::uint32_t index) {
        const std::uint8_t* edge_bytes = region + index * Routing2DCodec::MCAST_TREE_EDGE_BYTES;
        std::uint16_t edge;
        __builtin_memcpy(&edge, edge_bytes, sizeof(edge));
        return edge;
    }
};

// The shared multicast encoder updates a mutable byte map, while the device commits that completed
// map to L1 in words. Word backing provides aligned source loads; only a partial final word needs
// initialization because the encoder overwrites every active byte.
template <std::size_t CapacityBytes>
class Route2DMapStaging {
public:
    static constexpr std::size_t WORD_COUNT = (CapacityBytes + sizeof(std::uint32_t) - 1) / sizeof(std::uint32_t);

    explicit FORCE_INLINE Route2DMapStaging(std::uint32_t map_bytes) {
        const std::uint32_t tail_bytes = map_bytes % sizeof(std::uint32_t);
        if (tail_bytes != 0) {
            // The chunked copy reads the tail as a full word before storing only its valid low bytes.
            words[map_bytes / sizeof(std::uint32_t)] = 0;
        }
    }

    FORCE_INLINE std::uint8_t* bytes() { return reinterpret_cast<std::uint8_t*>(words); }
    FORCE_INLINE const std::uint32_t* word_data() const { return words; }

private:
    std::uint32_t words[WORD_COUNT];
};

// Copies the contiguous [Y | X] action map to the packet after encoding is complete. Full words use
// the naturally aligned route_buffer base; exact halfword/byte tails avoid touching following fields.
FORCE_INLINE void copy_2d_map_to_l1(
    volatile tt_l1_ptr std::uint8_t* output, const std::uint32_t* input_words, std::uint32_t map_bytes) {
    auto* output_words = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(output);
    const std::uint32_t full_words = map_bytes / sizeof(std::uint32_t);
    for (std::uint32_t i = 0; i < full_words; ++i) {
        output_words[i] = input_words[i];
    }

    const std::uint32_t tail_bytes = map_bytes % sizeof(std::uint32_t);
    if (tail_bytes != 0) {
        std::uint32_t tail = input_words[full_words];
        volatile tt_l1_ptr std::uint8_t* output_tail = output + full_words * sizeof(std::uint32_t);
        if (tail_bytes >= sizeof(std::uint16_t)) {
            *reinterpret_cast<volatile tt_l1_ptr std::uint16_t*>(output_tail) = static_cast<std::uint16_t>(tail);
            output_tail += sizeof(std::uint16_t);
            tail >>= sizeof(std::uint16_t) * 8;
        }
        if (tail_bytes & 1u) {
            *output_tail = static_cast<std::uint8_t>(tail);
        }
    }
}

// Streams widened action bytes into the packet's contiguous [Y | X] map. Full words start at the
// naturally aligned route_buffer base; the final 1-3 bytes use narrow stores so they cannot overwrite
// dst_start_node_id when the action-map size is not word-aligned.
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

// Installs destination dst_dev_id's action maps from the given route table. The destination's packed
// Y row becomes route_buffer[0..Y); its packed X row becomes route_buffer[Y..Y+X). A zero Y action
// transitions decode to X, whose destination slot carries LOCAL_DELIVER. Shared by worker unicast
// setup and intermesh landing.
inline void widen_2d_route_to_chip(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    const std::uint8_t* route_table,
    uint16_t dst_dev_id,
    uint8_t mesh_y_size,
    uint8_t mesh_x_size) {
    ASSERT(dst_dev_id < (uint32_t)mesh_y_size * mesh_x_size);
    ASSERT((uint32_t)mesh_y_size + mesh_x_size <= sizeof(packet_header->route_buffer));

    route_2d_detail::Route2DWordWriter output(packet_header->route_buffer);
    widen_2d_route(output, route_table, dst_dev_id, mesh_y_size, mesh_x_size);
}

}  // namespace tt::tt_fabric
