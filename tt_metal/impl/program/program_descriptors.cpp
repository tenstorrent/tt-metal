// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tile.hpp>

#include "impl/buffers/semaphore.hpp"
#include "tt_stl/overloaded.hpp"

namespace tt::tt_metal {

TileDescriptor::TileDescriptor(const Tile& tile) :
    height(tile.get_height()), width(tile.get_width()), transpose(tile.get_transpose_of_faces()) {}

std::optional<uint32_t> ProgramDescriptor::find_available_semaphore_id(
    const CoreCoord& core, CoreType core_type) const {
    std::bitset<NUM_SEMAPHORES> used_semaphores;

    // check existing semaphores
    for (const auto& sem_desc : semaphores) {
        if (sem_desc.core_type == core_type && sem_desc.core_ranges.contains(core)) {
            used_semaphores.set(sem_desc.id);
        }
    }

    // find first available semaphore ID
    for (uint32_t i = 0; i < NUM_SEMAPHORES; i++) {
        if (!used_semaphores.test(i)) {
            return i;
        }
    }
    return std::nullopt;
}

}  // namespace tt::tt_metal
