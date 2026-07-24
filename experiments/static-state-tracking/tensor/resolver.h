// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// sst::tensor::Resolver
// ----------------------------------------------------------------------------
// Single point where Tile-level facts become Face-level facts (TileConfig). One
// specialisation: `Resolver<TileShape<Face, Nr, Nc, L>>` — pure `constexpr`.
// When `Shape` is known at the call site (always, in the SST experiment) the
// whole `TileConfig` collapses to inlined immediates.
// ----------------------------------------------------------------------------

#include <cstdint>

#include "face_shape.h"
#include "format_traits.h"
#include "tile_shape.h"

namespace sst::tensor {

// ----------------------------------------------------------------------------
// TileConfig — the resolved, flattened per-tile description the UNPACK / MATH /
// PACK recipes consume: per-face geometry (`face_r_dim`, `face_c_dim`), how many
// faces make up the tile (`num_faces`), and the data format.
//
// Fields are defaulted so `Tracked<TileConfig>{}` default-constructs.
// ----------------------------------------------------------------------------
struct TileConfig {
    uint32_t face_r_dim = 0;   ///< Rows in a single face (e.g. 16).
    uint32_t face_c_dim = 0;   ///< Cols in a single face (16).
    uint32_t num_faces = 0;    ///< Faces per tile = num_faces_r * num_faces_c.
    uint32_t data_format = 0;  ///< Underlying integer of DataFormat.

    constexpr bool operator==(const TileConfig& o) const {
        return face_r_dim == o.face_r_dim && face_c_dim == o.face_c_dim && num_faces == o.num_faces &&
               data_format == o.data_format;
    }
    constexpr bool operator!=(const TileConfig& o) const { return !(*this == o); }
};

// ----------------------------------------------------------------------------
// tile_size_bytes_from_tile_config — one tile's on-L1 footprint from a resolved
// TileConfig. `tile_hw` is the total datum count = face_r_dim*face_c_dim*num_faces.
// ----------------------------------------------------------------------------
constexpr inline uint32_t tile_size_bytes_from_tile_config(const TileConfig& tile_config) {
    const uint32_t tile_hw = tile_config.face_r_dim * tile_config.face_c_dim * tile_config.num_faces;
    return tile_size_bytes_of(static_cast<DataFormat>(tile_config.data_format), tile_hw);
}

// ----------------------------------------------------------------------------
// Primary template. Never used directly — every supported `Shape` (a
// `TileShape<…>`) hits the specialisation.
// ----------------------------------------------------------------------------
template <typename Shape>
struct Resolver {
    static constexpr TileConfig tile_config() {
        static_assert(
            sizeof(Shape) == 0,
            "sst::tensor::Resolver: no specialisation for this Shape type. "
            "Expected a TileShape<...> instantiation.");
        return TileConfig{};
    }
};

// ----------------------------------------------------------------------------
// Compile-time specialisation: typed `TileShape<Face, Nr, Nc, L>`. `tile_config()`
// folds to immediates at the call site.
// ----------------------------------------------------------------------------
template <typename Face, uint8_t Nr, uint8_t Nc, Layout L>
struct Resolver<TileShape<Face, Nr, Nc, L>> {
    using shape_t = TileShape<Face, Nr, Nc, L>;

    static constexpr TileConfig tile_config() {
        return TileConfig{
            /* face_r_dim  */ uint32_t(Face::face_r_dim),
            /* face_c_dim  */ uint32_t(Face::face_c_dim),
            /* num_faces   */ uint32_t(Nr) * uint32_t(Nc),
            /* data_format */ uint32_t(Face::data_format),
        };
    }
};

}  // namespace sst::tensor
