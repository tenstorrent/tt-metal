// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "noc/att/att.h"

// Quasar's small Aether GRID_2x3/GRID_2x3_DISPATCH bring-up map. This is the
// map implemented by firmware/datamover/perf_testing_lib/include/aether_utils.h,
// extended with selectors for the top-row NOC2AXI/dispatch tiles.
//
// Data only. Shared address construction lives in att_address_map.h.
namespace quasar_aether_2x3_att_config {

// First 64 bits of the source aether_utils.h MD5.
constexpr uint64_t MAP_SIGNATURE = 0x43eaaf2bd7930da9ull;

// tt-metal's checked-in Aether descriptor exposes workers at (0,1),(1,1).
constexpr uint32_t ATT_WORKER_API_ORIGIN_X = 0;
constexpr uint32_t ATT_WORKER_API_ORIGIN_Y = 1;
constexpr uint32_t ATT_WORKER_GRID_X = 2;
constexpr uint32_t ATT_WORKER_GRID_Y = 1;

constexpr uint32_t ATT_TILE_GRID_X = 2;
constexpr uint32_t ATT_TILE_GRID_Y = 3;

constexpr uint8_t ATT_WORKER_SELECTORS[] = {0, 1};

// Array index uses tt-metal/UMD-visible y*2+x. Selectors 0..3 match
// aether_utils.h. The UMD descriptor exposes dispatch at (1,2), while the
// 2x3_DISPATCH RTL places it at (0,2), so the top-row selector order provides
// that alias explicitly.
constexpr uint8_t ATT_TILE_SELECTORS[] = {
    2,
    3,  // y=0: DRAM
    0,
    1,  // y=1: Tensix
    5,
    4,  // y=2 UMD view: NOC2AXI, dispatch
};

// Matches Aether::configure_aether_dram(GRID_2x3): bank 0 targets (0,0)
// through selector 2 and bank 1 targets (1,0) through selector 3.
constexpr uint8_t ATT_DRAM_SELECTORS[] = {2, 3};

constexpr uint64_t ATT_LOCAL_SCRATCH_OFFSET = 0;

constexpr noc_att::Window LOCAL_WINDOW{
    .compare = 0,
    .mask_bits = 36,
    .endpoint_shift = 26,
    .endpoint_size = 10,
    .endpoint_table_offset = 0,
    .translate_address = false,
};  // mask-table slot 0

constexpr noc_att::Window REMOTE_WINDOW{
    .compare = 0x1000000000ull,
    .mask_bits = 36,
    .endpoint_shift = 26,
    .endpoint_size = 10,
    .endpoint_table_offset = 1,
    .translate_address = true,
};  // mask-table slot 1, BAR/rebase 0

constexpr noc_att::Window WORKER_WINDOW = REMOTE_WINDOW;
constexpr noc_att::Window DRAM_WINDOW = REMOTE_WINDOW;
constexpr noc_att::Window TILE_WINDOW = REMOTE_WINDOW;
constexpr noc_att::Window WINDOWS[] = {LOCAL_WINDOW, REMOTE_WINDOW};

}  // namespace quasar_aether_2x3_att_config
