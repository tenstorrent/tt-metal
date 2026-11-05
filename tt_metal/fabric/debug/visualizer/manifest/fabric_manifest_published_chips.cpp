// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_published_chips.hpp"

#include <utility>

#include <tt_stl/assert.hpp>

namespace tt::tt_fabric::manifest {

void PublishedChips::publish(ChipId physical_chip_id, Chip chip, uint32_t num_initialized_routers) {
    TT_FATAL(
        physical_chip_id >= 0 && static_cast<size_t>(physical_chip_id) < chips_by_physical_id_.size(),
        "Fabric manifest: chip {} is past the {} chips the builder context holds",
        physical_chip_id,
        chips_by_physical_id_.size());
    auto& published = chips_by_physical_id_[physical_chip_id];
    TT_FATAL(!published.has_value(), "Fabric manifest: chip {} was already published", physical_chip_id);
    TT_FATAL(
        chip.routers.size() == num_initialized_routers,
        "Fabric manifest: chip {} published {} routers, but the builder initialized {}",
        physical_chip_id,
        chip.routers.size(),
        num_initialized_routers);
    published = std::move(chip);
}

bool PublishedChips::has(ChipId physical_chip_id) const {
    return physical_chip_id >= 0 && static_cast<size_t>(physical_chip_id) < chips_by_physical_id_.size() &&
           chips_by_physical_id_[physical_chip_id].has_value();
}

const Chip& PublishedChips::get(ChipId physical_chip_id) const {
    TT_FATAL(has(physical_chip_id), "Fabric manifest: chip {} was not published", physical_chip_id);
    return *chips_by_physical_id_[physical_chip_id];
}

}  // namespace tt::tt_fabric::manifest
