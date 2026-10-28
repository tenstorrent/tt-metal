// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Dimensions of the default 32x32 tile and its 16x16 faces, for device (kernel) code.
//
// This is the device-side counterpart of the host header tt-metalium/constants.hpp. The two files are
// deliberately separate: one is compiled by the host compiler, the other by the RISC-V kernel compiler,
// and neither may include the other. Kernels that must work with other tile shapes should take the tile
// dimensions as compile-time or runtime arguments instead of using these values.

namespace tt::constants {

constexpr std::uint32_t TILE_HEIGHT = 32;
constexpr std::uint32_t TILE_WIDTH = 32;
constexpr std::uint32_t TILE_HW = TILE_WIDTH * TILE_HEIGHT;
constexpr std::uint32_t FACE_HEIGHT = 16;
constexpr std::uint32_t FACE_WIDTH = 16;
constexpr std::uint32_t FACE_HW = FACE_WIDTH * FACE_HEIGHT;

}  // namespace tt::constants
