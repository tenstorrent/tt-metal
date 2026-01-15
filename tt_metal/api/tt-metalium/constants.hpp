// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::constants {

using std::uint32_t;

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t TILE_HW = TILE_WIDTH * TILE_HEIGHT;
constexpr uint32_t FACE_HEIGHT = 16;
constexpr uint32_t FACE_WIDTH = 16;
constexpr uint32_t FACE_HW = FACE_WIDTH * FACE_HEIGHT;
constexpr uint32_t BFLOAT8_B_TILE_HW = TILE_HW + 64;  // Value is in bytes - 1B per BFP8 datum + 4x16B exponent section
constexpr uint32_t BFLOAT4_B_TILE_HW =
    (TILE_HW / 2) + 64;  // Value is in bytes - 0.5B per BFP4 datum + 4x16B exponent section

}  // namespace tt::constants
