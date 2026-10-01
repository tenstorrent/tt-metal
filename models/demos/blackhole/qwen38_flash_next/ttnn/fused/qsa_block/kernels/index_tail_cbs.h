// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// CB indices, semaphores and tile geometry shared by the index_tail kernels (must match qsa_block/__init__.py).

#pragma once

#include <cstdint>

#include "tile_rows.h"

namespace index_tail {
constexpr uint32_t CB_X = 0, CB_SCALER = 1, CB_EPS = 2, CB_GAMMA_Q = 3, CB_XMM2 = 4, CB_EX2 = 5, CB_EX2PE = 6,
                   CB_FUSION = 7;
constexpr uint32_t CB_NQ = 8, CB_GAMMA_K = 9, CB_RING = 10, CB_ONES = 11, CB_RINGC = 12, CB_RINGW = 13,
                   CB_SCALER_RING = 14;
constexpr uint32_t CB_POOLED = 15, CB_NK = 16;
constexpr uint32_t CB_IN_Q = 17, CB_ROT_Q = 18, CB_COS = 19, CB_SIN = 20, CB_SCALAR = 21, CB_ROT_NEG = 22, CB_XCOS = 23;
constexpr uint32_t CB_RSIN = 24, CB_OUT_Q = 25, CB_IN_K = 26, CB_ROT_K = 27, CB_BCOS = 28, CB_BSIN = 29, CB_OUT_K = 30;
constexpr uint32_t CB_RAW = 31, CB_POS_R = 32, CB_POS_W = 33, CB_POSCR = 34,
                   CB_POSCW = 35;  // POSC*: the read_positions scratch
constexpr uint32_t SEM_Q = 0, SEM_K = 1, SEM_READY = 2;
constexpr uint32_t HEAD_TILES = 4, ROPE_TILES = 2;
using tile_rows::chunk_offset;
using tile_rows::copy_tile_row;
using tile_rows::ROW_BYTES;
using tile_rows::TILE_BYTES;
}  // namespace index_tail
