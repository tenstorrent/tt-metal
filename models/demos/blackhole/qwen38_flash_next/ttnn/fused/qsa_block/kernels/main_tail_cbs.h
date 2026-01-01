// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// CB indices, semaphores and geometry shared by the main_tail kernels (must match qsa_block/__init__.py).

#pragma once

#include <cstdint>

#include "tile_rows.h"

namespace main_tail {
constexpr uint32_t cb_x = 0, cb_scaler = 1, cb_eps = 2, cb_gamma = 3, cb_xmm2 = 4, cb_ex2 = 5, cb_ex2pe = 6,
                   cb_fusion = 7;
constexpr uint32_t cb_n = 8, cb_zero = 9;
constexpr uint32_t cb_in = 10, cb_rot = 11, cb_cos = 12, cb_sin = 13, cb_scalar = 14, cb_rot_neg = 15, cb_xcos = 16;
constexpr uint32_t cb_rsin = 17, cb_out = 18;
constexpr uint32_t cb_stg = 19, cb_ones = 20, cb_stgc = 21, cb_stgw = 22, cb_rm = 23, cb_pack = 24, cb_pos_r = 25,
                   cb_pos_w = 26, cb_poscr = 27, cb_poscw = 28;
constexpr uint32_t rope_semaphore_id = 0, key_rotated_semaphore_id = 1, key_normalized_semaphore_id = 2;
constexpr uint32_t head_tiles = 8, rope_tiles = 2, pack_tiles = 16, query_row_bytes = 1024, head_row_bytes = 512;
constexpr uint32_t query_heads = 32, local_heads = 6;
using tile_rows::chunk_offset;
using tile_rows::copy_tile_row;
using tile_rows::half_row_bytes;
using tile_rows::tile_bytes;
}  // namespace main_tail
