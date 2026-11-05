// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

namespace tt::tt_fabric::manifest {

// Each chip's routers and chip-wide facts, published by the chip's FabricBuilder once its kernels are created and read
// by the topology pass when the manifest is written. The builder context owns it, since it outlives each FabricBuilder.
class PublishedChips {
public:
    explicit PublishedChips(size_t num_chips) : chips_by_physical_id_(num_chips) {}

    // Publishes `chip`, whose routers must be every router its builder initialized.
    void publish(ChipId physical_chip_id, Chip chip, uint32_t num_initialized_routers);
    bool has(ChipId physical_chip_id) const;
    const Chip& get(ChipId physical_chip_id) const;

private:
    std::vector<std::optional<Chip>> chips_by_physical_id_;
};

}  // namespace tt::tt_fabric::manifest
