// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// harness/targets/x280_tile.hpp -- thin wrapper around a resident-idle X280
// tile used by the characterization suite. Owns nothing: binds the
// X280Tile* + TileId + X280Runtime* triple as one value so the runner and
// per-test callers hold one member instead of three.

#ifndef X280_HARNESS_TARGETS_X280_TILE_HPP
#define X280_HARNESS_TARGETS_X280_TILE_HPP

#include "x280/device/x280_runtime.hpp"
#include "x280/device/x280_tile.hpp"

namespace x280::harness::targets {

struct X280TileHandle {
    x280::X280Runtime* runtime = nullptr;
    x280::X280Tile* tile = nullptr;
    x280::TileId id{};

    x280::X280Tile& operator*() const { return *tile; }
    x280::X280Tile* operator->() const { return tile; }
    bool valid() const { return runtime != nullptr && tile != nullptr; }
};

}  // namespace x280::harness::targets

#endif  // X280_HARNESS_TARGETS_X280_TILE_HPP
