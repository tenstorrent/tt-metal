// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "noc/att/att.h"

// Data only. Shared address construction lives in att_address_map.h.
namespace grendel_qsr1_att_config {

// First 64 bits of the generated tile-NIU payload MD5.
constexpr uint64_t MAP_SIGNATURE = 0x71d0d0cedd8eb875ull;

constexpr uint32_t ATT_WORKER_API_ORIGIN_X = 2;
constexpr uint32_t ATT_WORKER_API_ORIGIN_Y = 2;
constexpr uint32_t ATT_WORKER_GRID_X = 8;
constexpr uint32_t ATT_WORKER_GRID_Y = 4;

constexpr uint32_t ATT_TILE_GRID_X = 10;
constexpr uint32_t ATT_TILE_GRID_Y = 6;

// Worker selectors are row-major logical coordinates.
// clang-format off
constexpr uint8_t ATT_WORKER_SELECTORS[] = {
     0,  1,  2,  3,  4,  5,  6,  7,
     8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31,
};

// qsr.s1 Config selectors are not row-major. Array index is physical y*10+x.
constexpr uint8_t ATT_TILE_SELECTORS[] = {
    59, 55, 54, 53, 52, 51, 50, 49, 48, 57,
    32,  0,  1,  2,  3,  4,  5,  6,  7, 47,
    33,  8,  9, 10, 11, 12, 13, 14, 15, 46,
    34, 16, 17, 18, 19, 20, 21, 22, 23, 45,
    35, 24, 25, 26, 27, 28, 29, 30, 31, 44,
    58, 36, 37, 38, 39, 40, 41, 42, 43, 56,
};

// Runtime DRAM banks are tile-major, then Mimir-major.
constexpr uint8_t ATT_DRAM_SELECTORS[] = {0, 1, 2, 3, 16, 17, 18, 19};
// clang-format on

constexpr uint64_t ATT_LOCAL_SCRATCH_OFFSET = 0;

constexpr noc_att::Window LOCAL_WINDOW{
    .compare = 0x100000ull,
    .mask_bits = 20,
    .endpoint_shift = 0,
    .endpoint_size = 0,
    .endpoint_table_offset = 256,
    .translate_address = false,
};  // mask-table slot 13

constexpr noc_att::Window WORKER_WINDOW{
    .compare = 0x10000000000ull,
    .mask_bits = 30,
    .endpoint_shift = 24,
    .endpoint_size = 6,
    .endpoint_table_offset = 128,
    .translate_address = true,
};  // mask-table slot 4

constexpr noc_att::Window DRAM_WINDOW{
    .compare = 0x1000000000000ull,
    .mask_bits = 38,
    .endpoint_shift = 33,
    .endpoint_size = 5,
    .endpoint_table_offset = 96,
    .translate_address = false,
};  // mask-table slot 5

constexpr noc_att::Window TILE_WINDOW{
    .compare = 0x1800000000ull,
    .mask_bits = 33,
    .endpoint_shift = 27,
    .endpoint_size = 6,
    .endpoint_table_offset = 256,
    .translate_address = true,
};  // mask-table slot 14

constexpr noc_att::Window WINDOWS[] = {LOCAL_WINDOW, WORKER_WINDOW, DRAM_WINDOW, TILE_WINDOW};

}  // namespace grendel_qsr1_att_config
