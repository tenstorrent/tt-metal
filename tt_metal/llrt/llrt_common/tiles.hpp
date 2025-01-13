// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt {

namespace tiles_test {

namespace tiles_types {
using TwoBInt = uint16_t;
}  // namespace tiles_types

using TileIndex = std::uint32_t;
using TileSize = std::uint32_t;

inline int get_src_channel_id_no_offset_from_tile_index(int tile_index) { return tile_index % 8; }

inline int get_src_core_index_no_offset_from_tile_index(int tile_index, int num_of_cores) {
    return tile_index % num_of_cores;
}

inline int get_src_core_index_from_tile_index(int tile_index, int num_of_cores, int core_count_offset) {
    return get_src_core_index_no_offset_from_tile_index(tile_index + core_count_offset, num_of_cores);
}

}  // namespace tiles_test

}  // namespace tt
