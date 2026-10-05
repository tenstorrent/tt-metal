// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/tt-2xx/quasar/noc/att/att.h"
#include "internal/tt-2xx/quasar/noc/att/att_address.h"

/**
 * @file
 * @brief Horizon 2x3 (aether) map, for the IP whole or split into per-column
 * devices. It has quasar_aether_2x3's geometry: a translating LOCAL window at
 * 0x18_0000_0000 (mask slot 0, selector bits [31:26], entry 0 = the issuing
 * tile) and a REMOTE window at 0x10_0000_0000 (mask slot 1, endpoint table
 * from 1). There is no compare=0 window, so a raw address matches no window.
 *
 * Coordinates and selectors are in the device's frame, the coordinates of its
 * soc descriptor: the whole grid is 2x3, and a split device is 1x3 at x = 0 on
 * either column, so it uses selectors 0, 2 and 4. The build's ATT program
 * (tt-umd-simulators emu/horizon_split/att), which UMD replays before any DM
 * core starts, maps each device's selectors to its own tiles' NODE_IDs and
 * parks the rest.
 *
 * Data only. Address resolution over this data lives in noc/att/att_address.h.
 */
namespace horizon_2x3_att_config {

constexpr std::uint32_t ATT_WORKER_API_ORIGIN_X = 0;
constexpr std::uint32_t ATT_WORKER_API_ORIGIN_Y = 0;
constexpr std::uint32_t ATT_WORKER_GRID_X = 2;
constexpr std::uint32_t ATT_WORKER_GRID_Y = 1;

// Selectors: 0-1 Tensix (0,0),(1,0); 2-3 Dispatch (0,1),(1,1); 4-5 NOC2AXI
// (0,2),(1,2), which are also DRAM banks 0 and 1. Endpoint words are
// (y << 6) | x in the device's frame.
constexpr std::uint8_t ATT_WORKER_SELECTORS[] = {0, 1};
constexpr std::uint16_t ATT_WORKER_ENDPOINT_WORDS[] = {0x000, 0x001};
constexpr std::uint16_t ATT_FULL_TILE_ENDPOINT_WORDS[] = {0x000, 0x001, 0x040, 0x041, 0x080, 0x081};
constexpr std::uint8_t ATT_LOGICAL_DRAM_SELECTORS[] = {4, 5};

// quasar_aether_2x3's windows: selector bits [31:26] of the LOCAL base fold to
// 0, so base | local selects entry 0, which the ATT program sets to the tile.
constexpr noc_att::Window LOCAL_WINDOW{
    .compare = 0x1800000000ull,
    .mask_bits = 33,
    .endpoint_shift = 26,
    .endpoint_size = 6,
    .endpoint_table_offset = 0,
    .translate_address = true,
};  // mask-table slot 0

constexpr noc_att::Window REMOTE_WINDOW{
    .compare = 0x1000000000ull,
    .mask_bits = 32,
    .endpoint_shift = 26,
    .endpoint_size = 6,
    .endpoint_table_offset = 1,
    .translate_address = true,
};  // mask-table slot 1, BAR 0

// Every operand for this initiator's own L1 is LOCAL_WINDOW_BASE | local_address.
constexpr std::uint64_t LOCAL_WINDOW_BASE = LOCAL_WINDOW.make_address(/*selector*/ 0, /*local_address*/ 0);
static_assert(LOCAL_WINDOW_BASE == 0x1800000000ull);

inline constexpr noc_att::MapData::DispatchEntry DISPATCH_ENTRIES[] = {
    {.x = 0, .y = 1, .selector = 2, .window = noc_att::WindowClass::FullTile},
    {.x = 1, .y = 1, .selector = 3, .window = noc_att::WindowClass::FullTile},
};

// One remote window, so the Worker/Dram/FullTile roles all point at it and
// differ only in which selector table resolution consults.
inline constexpr noc_att::MapData MAP{
    .windows = {{noc_att::NO_WINDOW, REMOTE_WINDOW, REMOTE_WINDOW, REMOTE_WINDOW, LOCAL_WINDOW}},
    .local_window_class = noc_att::WindowClass::Local,  // translating local window, entry 0 = self
    .worker_origin_x = ATT_WORKER_API_ORIGIN_X,
    .worker_origin_y = ATT_WORKER_API_ORIGIN_Y,
    .worker_grid_x = ATT_WORKER_GRID_X,
    .worker_grid_y = ATT_WORKER_GRID_Y,
    .worker_selectors = {ATT_WORKER_SELECTORS},
    .worker_endpoint_words = {ATT_WORKER_ENDPOINT_WORDS},
    .full_tile_endpoint_words = {ATT_FULL_TILE_ENDPOINT_WORDS},
    .dram_selectors = {ATT_LOGICAL_DRAM_SELECTORS},
    .dispatch_entries = {DISPATCH_ENTRIES},
};

}  // namespace horizon_2x3_att_config
