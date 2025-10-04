// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tile.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <utility>

#include "core_coord.hpp"

namespace tt::tt_metal {

TileDescriptor::TileDescriptor(const Tile& tile) :
    height(tile.get_height()), width(tile.get_width()), transpose(tile.get_transpose_of_faces()) {}

uint32_t ProgramDescriptor::add_semaphore(CoreRangeSet core_ranges, uint32_t initial_value, CoreType core_type) {
    semaphores.emplace_back(core_type, std::move(core_ranges), initial_value);
    return semaphores.size() - 1;
}

}  // namespace tt::tt_metal
