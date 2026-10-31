// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host-only header. Device (kernel/firmware) code must use api/tile_constants.h from tt_metal/hw/inc instead:
// the two sides are compiled by different compilers and must not share this file.
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#error "tt-metalium/constants.hpp is host-only; include \"api/tile_constants.h\" in device code"
#endif

#include <cstdint>

namespace tt::constants {

using std::uint32_t;

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t TILE_HW = TILE_WIDTH * TILE_HEIGHT;
constexpr uint32_t FACE_HEIGHT = 16;
constexpr uint32_t FACE_WIDTH = 16;
constexpr uint32_t FACE_HW = FACE_WIDTH * FACE_HEIGHT;

}  // namespace tt::constants
