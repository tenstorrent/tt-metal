// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <board/board.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::scaleout_tools {

inline AsicChannel make_asic_channel(uint32_t asic_location, uint8_t src_chan) {
    return AsicChannel{asic_location, ChanId{src_chan}};
}

inline std::optional<Port> try_get_port(const Board& board, uint32_t asic_location, uint8_t src_chan) {
    try {
        return board.get_port_for_asic_channel(make_asic_channel(asic_location, src_chan));
    } catch (const std::runtime_error&) {
        return std::nullopt;
    }
}

inline std::optional<Port> try_get_port(BoardType board_type, uint32_t asic_location, uint8_t src_chan) {
    return try_get_port(create_board(board_type), asic_location, src_chan);
}

inline PortType resolve_port_type(BoardType board_type, uint32_t asic_location, uint8_t src_chan) {
    auto port = try_get_port(board_type, asic_location, src_chan);
    return port.has_value() ? port->port_type : PortType::UNKNOWN;
}

}  // namespace tt::scaleout_tools
