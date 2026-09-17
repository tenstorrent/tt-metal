// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Tile/face-geometry precedence — the emule twin of jit_build/jit_build_options.cpp's
// set_cb_data_fmt_tile_and_face_geometry. Must stay bit-for-bit consistent with silicon's
// precedence: an explicit unpack FaceGeometry > the CB's Tile > the full-tile default.

#include <optional>

#include <tt-metalium/face_geometry.hpp>
#include <tt-metalium/tile.hpp>

namespace tt::tt_metal::emule {

// Mirrors the per-CB descriptor rule in jit_build/jit_build_options.cpp: silicon bakes this
// geometry into chlkc_descriptors.h, so the emulated kernel has to be handed the same answer.
struct ResolvedTileGeometry {
    Tile tile;  // effective tile: an explicit FaceGeometry substitutes its own
    uint32_t num_faces = tt::constants::TILE_HW / tt::constants::FACE_HW;
    uint32_t face_r_dim = tt::constants::FACE_HEIGHT;
    uint32_t partial_face = 0;
    uint32_t narrow_tile = 0;
};

// Precedence: an explicit unpack FaceGeometry wins over the CB's Tile, which wins over the
// full-tile default.
ResolvedTileGeometry resolve_tile_geometry(
    const std::optional<Tile>& tile, const std::optional<FaceGeometry>& unpack_face_geometry);

}  // namespace tt::tt_metal::emule
