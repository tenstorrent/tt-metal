// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// CB indices, semaphores and tile geometry shared by the index_tail kernels (must match qsa_block/__init__.py).

#pragma once

#include <cstdint>

#include "tile_rows.h"

namespace index_tail {
constexpr uint32_t cb_x = 0, cb_scaler = 1, cb_eps = 2, cb_gamma_q = 3, cb_xmm2 = 4, cb_ex2 = 5, cb_ex2pe = 6,
                   cb_fusion = 7;
constexpr uint32_t cb_nq = 8, cb_gamma_k = 9, cb_ring = 10, cb_ones = 11, cb_ringc = 12, cb_ringw = 13,
                   cb_scaler_ring = 14;
constexpr uint32_t cb_pooled = 15, cb_nk = 16;
constexpr uint32_t cb_in_q = 17, cb_rot_q = 18, cb_cos = 19, cb_sin = 20, cb_scalar = 21, cb_rot_neg = 22, cb_xcos = 23;
constexpr uint32_t cb_rsin = 24, cb_out_q = 25, cb_in_k = 26, cb_rot_k = 27, cb_bcos = 28, cb_bsin = 29, cb_out_k = 30;
constexpr uint32_t cb_raw = 31, cb_pos_r = 32, cb_pos_w = 33, cb_poscr = 34,
                   cb_poscw = 35;  // cb_posc*: the read_positions scratch
constexpr uint32_t query_semaphore_id = 0, key_semaphore_id = 1, ready_semaphore_id = 2;
constexpr uint32_t head_tiles = 4, rope_tiles = 2;
using tile_rows::chunk_offset;
using tile_rows::copy_tile_row;
using tile_rows::half_row_bytes;
using tile_rows::tile_bytes;
}  // namespace index_tail
