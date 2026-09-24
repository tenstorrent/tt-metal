// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Overlays each socket's shm ring onto its arena slot in the region, so the NIC reads and
// writes the socket's own pages. MUST be built before HostRegion::provision() pins.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tt_metal/distributed/host_region.hpp"

namespace tt::tt_metal::experimental {

class RingAlias {
public:
    // One per socket, taken straight off its HDSocketDescriptor.
    struct Slot {
        std::string shm_name;
        uint64_t shm_size = 0;
        uint32_t data_offset = 0;
        uint32_t fifo_size = 0;
    };

    static std::unique_ptr<RingAlias> map(
        uint8_t* region_base, AliasArena arena, const std::vector<Slot>& slots, std::string& err);

    // Restores anonymous pages over each slot and clears the region's declarations.
    ~RingAlias();

    RingAlias(const RingAlias&) = delete;
    RingAlias& operator=(const RingAlias&) = delete;

    // Mapping base for `core`; add the descriptor's own offsets to reach the data region,
    // the counter word or the connector state.
    uint8_t* base(uint32_t core) const;
    uint32_t count() const;
    std::string describe() const;

private:
    RingAlias();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::experimental
