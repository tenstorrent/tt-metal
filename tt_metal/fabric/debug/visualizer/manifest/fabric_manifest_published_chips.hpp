// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "tt_metal/fabric/builder/connection_registry.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

namespace tt::tt_fabric::manifest {

// A chip as its FabricBuilder publishes it. The builders name routers by Ethernet channel, and the topology pass
// resolves those channels to the routers the model refers to, so the channels stay here rather than in the model.
struct PublishedChip {
    Chip chip;
    // The Ethernet channel of the router that leads the chip's local sync (KernelCreationContext::master_router_chan).
    uint32_t local_sync_master_eth_chan = 0;
    // Every connection between the chip's routers, as establish_connections_to_router recorded it.
    std::vector<RouterConnectionRecord> connections;
};

// Each chip's routers and chip-wide facts, published by the chip's FabricBuilder once its kernels are created and read
// by the topology pass when the manifest is written. The builder context owns it, since it outlives each FabricBuilder.
class PublishedChips {
public:
    explicit PublishedChips(size_t num_chips) : chips_by_physical_id_(num_chips) {}

    // Publishes `published`, whose chip's routers must be every router its builder initialized.
    void publish(ChipId physical_chip_id, PublishedChip published, uint32_t num_initialized_routers);
    bool has(ChipId physical_chip_id) const;
    const PublishedChip& get(ChipId physical_chip_id) const;

private:
    std::vector<std::optional<PublishedChip>> chips_by_physical_id_;
};

}  // namespace tt::tt_fabric::manifest
