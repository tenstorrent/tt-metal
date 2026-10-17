// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// harness/regions/release_guard.hpp -- guard words that show whether a Tensix release reached a
// core outside its selection.

#ifndef X280_HARNESS_REGIONS_RELEASE_GUARD_HPP
#define X280_HARNESS_REGIONS_RELEASE_GUARD_HPP

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "harness/metal_context.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace x280::harness::regions {

/** The word a guard core's release slot holds through a run; a release token
 * landing there would replace it. */
constexpr uint32_t kGuardSentinel = 0xA5A5A5A5u;

/** Every logical core of the mesh's compute grid outside selection, row by row. */
inline std::vector<tt::tt_metal::CoreCoord> guards_outside(
    const tt::tt_metal::distributed::MeshDevice& mesh, const tt::tt_metal::CoreRangeSet& selection) {
    const auto grid = mesh.compute_with_storage_grid_size();
    std::vector<tt::tt_metal::CoreCoord> guards;
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            if (!selection.contains(tt::tt_metal::CoreCoord{x, y})) {
                guards.push_back(tt::tt_metal::CoreCoord{x, y});
            }
        }
    }
    return guards;
}

/** A guard core whose word no longer held kGuardSentinel, and the word it held. */
struct OverwrittenGuard {
    tt::tt_metal::CoreCoord core;
    uint32_t observed;
};

/** Reads addr back on each core in guards, after their sentinels were written;
 * returns those that lost kGuardSentinel, in guard order. Blocking. */
inline std::vector<OverwrittenGuard> overwritten_guards(
    const tt::tt_metal::distributed::MeshDevice& mesh,
    const std::vector<tt::tt_metal::CoreCoord>& guards,
    uint32_t addr) {
    auto& c = x280::harness::cluster();
    std::vector<OverwrittenGuard> out;
    for (const auto& g : guards) {
        uint32_t observed = 0u;
        c.read_core(
            &observed, sizeof(observed), tt_cxy_pair(mesh.build_id(), mesh.worker_core_from_logical_core(g)), addr);
        if (observed != kGuardSentinel) {
            out.push_back({g, observed});
        }
    }
    return out;
}

/** Writes kGuardSentinel to L1 address addr on each logical Tensix core in
 * guards and returns once every one reads back, so no write is still in flight
 * behind a later host read such as a clock sample. Throws std::runtime_error
 * naming the first core whose sentinel did not land. The caller owns that word
 * on those cores for the run: no kernel or buffer may use it. */
inline void write_guards(
    const tt::tt_metal::distributed::MeshDevice& mesh,
    const std::vector<tt::tt_metal::CoreCoord>& guards,
    uint32_t addr) {
    auto& c = x280::harness::cluster();
    for (const auto& g : guards) {
        c.write_core(
            &kGuardSentinel,
            sizeof(kGuardSentinel),
            tt_cxy_pair(mesh.build_id(), mesh.worker_core_from_logical_core(g)),
            addr);
    }
    const auto missing = overwritten_guards(mesh, guards, addr);
    if (!missing.empty()) {
        throw std::runtime_error("guard sentinel did not land on logical core " + missing.front().core.str());
    }
}

}  // namespace x280::harness::regions

#endif  // X280_HARNESS_REGIONS_RELEASE_GUARD_HPP
