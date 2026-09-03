// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/face_geometry.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_metal {

// Host-side metadata needed to specialize an LLK operand. ProgramSpec resolution
// materializes omitted Tile and FaceGeometry values before storing this object.
// Absence of LLK metadata is represented by std::optional<LLKMetadata>.
struct LLKMetadata {
    DataFormat format;
    Tile tile;
    FaceGeometry face_geometry;
};

}  // namespace tt::tt_metal
