// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// CB indices, semaphores and geometry shared by the main_tail kernels (must match qsa_block/__init__.py).

#pragma once

#include <cstdint>

#include "tile_rows.h"

namespace main_tail {
constexpr uint32_t CB_X = 0, CB_SCALER = 1, CB_EPS = 2, CB_GAMMA = 3, CB_XMM2 = 4, CB_EX2 = 5, CB_EX2PE = 6,
                   CB_FUSION = 7;
constexpr uint32_t CB_N = 8, CB_ZERO = 9;
constexpr uint32_t CB_IN = 10, CB_ROT = 11, CB_COS = 12, CB_SIN = 13, CB_SCALAR = 14, CB_ROT_NEG = 15, CB_XCOS = 16;
constexpr uint32_t CB_RSIN = 17, CB_OUT = 18;
constexpr uint32_t CB_STG = 19, CB_ONES = 20, CB_STGC = 21, CB_STGW = 22, CB_RM = 23, CB_PACK = 24, CB_POS_R = 25,
                   CB_POS_W = 26, CB_POSCR = 27, CB_POSCW = 28;
constexpr uint32_t SEM_ROPE = 0, SEM_KROT = 1, SEM_KNORM = 2;
constexpr uint32_t HEAD_TILES = 8, ROPE_TILES = 2, PACK_TILES = 16, QUERY_ROW_BYTES = 1024, HEAD_ROW_BYTES = 512;
constexpr uint32_t QUERY_HEADS = 32, LOCAL_HEADS = 6;
using tile_rows::chunk_offset;
using tile_rows::copy_tile_row;
using tile_rows::ROW_BYTES;
using tile_rows::TILE_BYTES;
}  // namespace main_tail
