// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_packet_header.hpp"

#include <algorithm>
#include <bit>
#include <optional>
#include <string>
#include <vector>

#include <hostdevcommon/fabric_common.h>
#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_interpret.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_routing_table.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_paths.hpp"

namespace tt::tt_fabric::decode {

namespace {

// `width` bits of `value` from bit `offset`. Zero past bit 63, so a layout of any size never shifts out of range.
uint64_t read_bits(uint64_t value, uint32_t offset, uint32_t width) {
    if (offset >= 64) {
        return 0;
    }
    const uint64_t shifted = value >> offset;
    return width >= 64 ? shifted : shifted & ((uint64_t{1} << width) - 1);
}

// A command's flush flag, read from its byte rather than as a bool, since a header may hold any byte there. The router
// flushes for any nonzero byte.
bool read_flush(const bool& flush) { return std::bit_cast<uint8_t>(flush) != 0; }

json make_noc_coordinates_json(uint32_t x, uint32_t y) {
    json out;
    out["x"] = x;
    out["y"] = y;
    return out;
}

// { noc }: one NOC address, as for a unicast write or read.
json make_unicast_json(const NocUnicastCommandHeader& unicast, const manifest::NocAddressLayout& layout) {
    json out;
    out["noc"] = make_noc_address_json(unicast.noc_address, layout);
    return out;
}

// The router writes the chunks in order: each but the last a write of its chunk_size, then the last by its encoding
// (the 2 bits at chunk_count - 1): a write of the rest of the payload (size_bytes null), or a semaphore increment by
// its chunk_size. The encodings of the other chunks aren't read. A count past NOC_SCATTER_WRITE_MAX_CHUNKS is cut
// there.
json make_scatter_write_json(const NocUnicastScatterCommandHeader& scatter, const manifest::NocAddressLayout& layout) {
    constexpr uint32_t k_num_chunk_sizes = NOC_SCATTER_WRITE_MAX_CHUNKS - 1;
    const uint32_t chunk_count = std::min<uint32_t>(scatter.chunk_count, NOC_SCATTER_WRITE_MAX_CHUNKS);
    json chunks = json::array();
    json last_chunk_encoding = nullptr;
    for (uint32_t chunk = 0; chunk + 1 < chunk_count; ++chunk) {
        json write;
        write["noc"] = make_noc_address_json(scatter.noc_address[chunk], layout);
        write["size_bytes"] = scatter.chunk_size[chunk];
        chunks.push_back(std::move(write));
    }
    if (chunk_count > 0) {
        const uint32_t last = chunk_count - 1;
        const auto encoding = static_cast<NocScatterWriteChunkEncoding>(
            (scatter.chunk_encoding >> (last * NOC_SCATTER_CHUNK_ENCODING_BITS)) & NOC_SCATTER_CHUNK_ENCODING_MASK);
        json last_chunk;
        last_chunk["noc"] = make_noc_address_json(scatter.noc_address[last], layout);
        switch (encoding) {
            case CHUNK_ENCODING_NOP: break;
            case CHUNK_ENCODING_UNICAST_WRITE: last_chunk["size_bytes"] = nullptr; break;
            case CHUNK_ENCODING_SEMINC_NO_FLUSH:
            case CHUNK_ENCODING_SEMINC_FLUSH:
                // The increment is the last chunk's chunk_size, which a fourth chunk has none of.
                last_chunk["increment"] = last < k_num_chunk_sizes ? json(scatter.chunk_size[last]) : json(nullptr);
                break;
        }
        chunks.push_back(std::move(last_chunk));
        last_chunk_encoding = detail::make_enum_json(encoding);
    }
    json out;
    out["chunks"] = std::move(chunks);
    out["last_chunk"] = std::move(last_chunk_encoding);
    return out;
}

// The targets of the chips the packet still writes, in hop order: from chip_idx to num_chips, counts[chip] addresses
// each, starting at write_idx, as each writing router advances the two. Indices past the arrays are cut there.
json make_sparse_multicast_write_json(
    const NocSparseMulticastWriteCommandHeader& sparse, const manifest::NocAddressLayout& layout) {
    const uint32_t num_chips = std::min<uint32_t>(sparse.num_chips, NOC_SPARSE_MCAST_WRITE_MAX_DESTS);
    uint32_t address_index = sparse.write_idx;
    json writes = json::array();
    for (uint32_t chip = sparse.chip_idx; chip < num_chips; ++chip) {
        json chip_writes = json::array();
        for (uint32_t write = 0; write < sparse.counts[chip] && address_index < NOC_SPARSE_MCAST_WRITE_MAX_DESTS;
             ++write, ++address_index) {
            chip_writes.push_back(make_noc_address_json(sparse.noc_address[address_index], layout));
        }
        writes.push_back(std::move(chip_writes));
    }
    json out;
    out["writes"] = std::move(writes);
    return out;
}

// { address, start, size }: a NOC multicast's rectangle, as the header holds it.
json make_multicast_json(uint32_t address, uint8_t x_start, uint8_t y_start, uint8_t size_x, uint8_t size_y) {
    json out;
    out["address"] = address;
    out["start"] = make_noc_coordinates_json(x_start, y_start);
    out["size"] = make_noc_coordinates_json(size_x, size_y);
    return out;
}

// The key a multicast hop count is written under. The switch has no default, so -Wswitch flags a direction added to
// eth_chan_directions that isn't named here; the TT_FATAL after it catches COUNT or an out-of-range value.
const char* name_mcast_direction(eth_chan_directions direction) {
    switch (direction) {
        case eth_chan_directions::EAST: return "east";
        case eth_chan_directions::WEST: return "west";
        case eth_chan_directions::NORTH: return "north";
        case eth_chan_directions::SOUTH: return "south";
        case eth_chan_directions::Z: return "z";
        case eth_chan_directions::COUNT: break;
    }
    TT_FATAL(
        false,
        "DRIFT CAUGHT IN FABRIC VISUALIZER - DECODE: No multicast name for eth_chan_directions value {}",
        static_cast<uint32_t>(direction));
}

}  // namespace

json make_packet_header_json(uint16_t payload_size_bytes, NocSendType noc_send_type, uint8_t src_ch_id) {
    json out;
    out["noc_send_type"] = detail::make_enum_json(noc_send_type);
    out["payload_size_bytes"] = payload_size_bytes;
    out["src_ch_id"] = src_ch_id;
    return out;
}

json make_noc_address_json(uint64_t noc_address, const manifest::NocAddressLayout& layout) {
    json out;
    out["x"] = read_bits(noc_address, layout.local_bits, layout.node_id_bits);
    out["y"] = read_bits(noc_address, layout.local_bits + layout.node_id_bits, layout.node_id_bits);
    out["address"] = read_bits(noc_address, 0, layout.local_bits);
    return out;
}

json make_noc_command_json(
    const NocCommandFields& command_fields, NocSendType noc_send_type, const InterpretContext& context) {
    const manifest::NocAddressLayout& layout = context.arch.noc_address;
    switch (noc_send_type) {
        case NOC_UNICAST_WRITE: return make_unicast_json(command_fields.unicast_write, layout);
        case NOC_UNICAST_READ: return make_unicast_json(command_fields.unicast_read, layout);
        case NOC_UNICAST_INLINE_WRITE: {
            const NocUnicastInlineWriteCommandHeader& inline_write = command_fields.unicast_inline_write;
            json out;
            out["noc"] = make_noc_address_json(inline_write.noc_address, layout);
            out["value"] = inline_write.value;
            return out;
        }
        case NOC_UNICAST_ATOMIC_INC: {
            const NocUnicastAtomicIncCommandHeader& atomic_inc = command_fields.unicast_seminc;
            json out;
            out["noc"] = make_noc_address_json(atomic_inc.noc_address, layout);
            out["increment"] = atomic_inc.val;
            out["flush"] = read_flush(atomic_inc.flush);
            return out;
        }
        case NOC_FUSED_UNICAST_ATOMIC_INC: {
            const NocUnicastAtomicIncFusedCommandHeader& fused = command_fields.unicast_seminc_fused;
            json out;
            out["noc"] = make_noc_address_json(fused.noc_address, layout);
            out["semaphore_noc"] = make_noc_address_json(fused.semaphore_noc_address, layout);
            out["increment"] = fused.val;
            out["flush"] = read_flush(fused.flush);
            return out;
        }
        case NOC_UNICAST_SCATTER_WRITE: return make_scatter_write_json(command_fields.unicast_scatter_write, layout);
        case NOC_SPARSE_MCAST_WRITE: return make_sparse_multicast_write_json(command_fields.sparse_mcast_write, layout);
        case NOC_MULTICAST_WRITE: {
            const NocMulticastCommandHeader& multicast = command_fields.mcast_write;
            return make_multicast_json(
                multicast.address,
                multicast.noc_x_start,
                multicast.noc_y_start,
                multicast.mcast_rect_size_x,
                multicast.mcast_rect_size_y);
        }
        case NOC_MULTICAST_ATOMIC_INC: {
            const NocMulticastAtomicIncCommandHeader& multicast = command_fields.mcast_seminc;
            json out = make_multicast_json(
                multicast.address, multicast.noc_x_start, multicast.noc_y_start, multicast.size_x, multicast.size_y);
            out["increment"] = multicast.val;
            return out;
        }
    }
    return json(nullptr);
}

std::vector<RoutingFieldsConstants::LowLatency::HopAction> read_1d_hop_actions(
    uint32_t routing_word, std::span<const uint32_t> extension_words) {
    using LowLatency = RoutingFieldsConstants::LowLatency;
    std::vector<LowLatency::HopAction> actions;
    uint32_t word = routing_word;
    size_t next_extension_word = 0;

    while (true) {
        // Process the next action.
        const auto action = static_cast<LowLatency::HopAction>(word & LowLatency::FIELD_MASK);
        actions.push_back(action);

        // End of the route.
        if ((action & LowLatency::FORWARD_ONLY) == 0) {
            break;
        }

        // Move to the next action, refilling the current word from the extension words if the current word has been
        // used completely.
        word >>= LowLatency::FIELD_WIDTH;
        if (word == 0 && next_extension_word < extension_words.size()) {
            word = extension_words[next_extension_word++];
        }
    }
    return actions;
}

json make_1d_route_json(std::span<const RoutingFieldsConstants::LowLatency::HopAction> actions) {
    json names = json::array();
    for (const RoutingFieldsConstants::LowLatency::HopAction action : actions) {
        names.push_back(detail::make_1d_hop_action_json(action));
    }
    json out;
    out["actions"] = std::move(names);
    return out;
}

json make_2d_action_json(uint8_t action) {
    static_assert(
        Routing2DCodec::ACTION_ETH_MASK == (1u << eth_chan_directions::COUNT) - 1,
        "DRIFT CAUGHT IN FABRIC VISUALIZER - DECODE: An action byte's low bits are one per eth_chan_directions "
        "direction");
    static_assert(
        Routing2DCodec::ACTION_LOCAL_DELIVER == 1u << eth_chan_directions::COUNT,
        "DRIFT CAUGHT IN FABRIC VISUALIZER - DECODE: Local delivery is the bit above the directions");
    json names = json::array();
    for (uint32_t bit = 0; bit < 8; ++bit) {
        // Skip bits that are not set in the action byte.
        if ((action & (1u << bit)) == 0) {
            continue;
        }
        std::optional<std::string> name;
        // Add the name of the direction if it is set in the action byte.
        if (bit < eth_chan_directions::COUNT) {
            name = find_enum_name(static_cast<eth_chan_directions>(bit));
        } else if ((1u << bit) == Routing2DCodec::ACTION_LOCAL_DELIVER) {  // local delivery special case
            name = "local";
        }
        // Just display the value if we don't have a name for it.
        names.push_back(name.value_or("bit" + std::to_string(bit)));
    }
    return names;
}

json make_2d_action_vocabulary_json() {
    json names = json::array();
    for (uint32_t bit = 0; bit < 8; ++bit) {
        names.push_back(make_2d_action_json(static_cast<uint8_t>(1u << bit)).front());
    }
    return names;
}

Route2DMaps split_2d_route_maps(std::span<const uint8_t> route_buffer, const manifest::Mesh& mesh) {
    // The Y map is the first mesh_y_size bytes and the X map the next mesh_x_size, cut at the end of the buffer.
    const std::vector<uint32_t>& shape = mesh.shape;
    const size_t y_size = shape.size() == 2 ? std::min<size_t>(shape[0], route_buffer.size()) : 0;
    const size_t x_size = shape.size() == 2 ? std::min<size_t>(shape[1], route_buffer.size() - y_size) : 0;
    return Route2DMaps{.y = route_buffer.subspan(0, y_size), .x = route_buffer.subspan(y_size, x_size)};
}

json make_2d_route_json(const Route2DMaps& maps, FabricNodeId dst_start, const std::array<uint16_t, 4>& mcast_params) {
    // Goes through every action in `map` and converts it to a JSON object.
    const auto make_map_json = [](std::span<const uint8_t> map) {
        json out = json::array();
        for (const uint8_t action : map) {
            out.push_back(make_2d_action_json(action));
        }
        return out;
    };

    // The two maps, each action byte as names.
    json out;
    out["y"] = make_map_json(maps.y);
    out["x"] = make_map_json(maps.x);

    out["dst"] = manifest::chip_path(dst_start);

    // Handle multicast parameters.
    if (std::ranges::all_of(mcast_params, [](uint16_t hops) { return hops == 0; })) {
        out["mcast"] = nullptr;
    } else {
        json mcast;
        for (uint32_t direction = 0; direction < mcast_params.size(); ++direction) {
            mcast[name_mcast_direction(static_cast<eth_chan_directions>(direction))] = mcast_params[direction];
        }
        out["mcast"] = std::move(mcast);
    }
    return out;
}

}  // namespace tt::tt_fabric::decode
