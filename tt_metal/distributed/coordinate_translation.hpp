// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>

#include "umd/device/types/cluster_descriptor_types.h"
#include <mesh_device_view.hpp>

namespace tt::tt_metal::distributed {

// TODO: Consider conversion to StrongType instead of alias
using LogicalCoordinate = Coordinate;
using PhysicalCoordinate = eth_coord_t;
using CoordinateTranslationMap = std::unordered_map<LogicalCoordinate, PhysicalCoordinate>;

// Returns a translation map between logical coordinates in logical 2D space
// to the physical coordinates as defined by the UMD layer.
std::pair<CoordinateTranslationMap, MeshShape> get_system_mesh_coordinate_translation_map();

}  // namespace tt::tt_metal::distributed
