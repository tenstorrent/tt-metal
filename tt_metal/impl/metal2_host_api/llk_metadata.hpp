// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_metal {

// Host-side format and tile used to emit binding_details::LLKMetadata onto a binding token.
// Face layout is read from the tile.
struct LLKMetadata {
    DataFormat format;
    Tile tile;
};

}  // namespace tt::tt_metal
