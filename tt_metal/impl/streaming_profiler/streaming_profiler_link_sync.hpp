// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include "hostdev/streaming_profiler_common.h"

namespace tt {
class Cluster;
}

// Which eth links of a chip pair carry the device-to-device link sync and which end of each sends. The device side
// is tools/profiler/sync/eth_ptp_link.hpp on the constants of hostdev/streaming_profiler_common.h; without fabric
// the profiler runs the ends as resident kernels, with fabric the routers on the chosen links run them
// (fabric_erisc_router.cpp, LINK_SYNC_ROLE).
namespace tt::tt_metal::streaming_profiler::link_sync {

enum class Role : uint32_t { None = 0, Sender = 1, Receiver = 2 };
// TT_METAL_STREAMING_PROFILER_LINK_SYNC=0 leaves the link sync out of a profiler session: no links planned, every
// router's role None. The rest of the profiler runs as usual; it is how the sync's own cost is measured.
bool enabled();

struct Link {
    uint32_t chip_a = 0, chip_b = 0;  // chip_a < chip_b; chip_a's end sends
    CoreCoord eth_a, eth_b;           // logical eth cores
};

// Every eligible link between two connected chips, in the cluster's order: the lower chip's eth cores connected to
// the higher and the cores they connect to; with fabric on, only links whose two cores hold routers. The sync runs
// over all of them and averages a pair's links, so their path asymmetries average too.
std::vector<Link> links_between(const tt::Cluster& cluster, uint32_t chip_x, uint32_t chip_y);
// What the router on this eth core does for the sync.
Role role_of(const tt::Cluster& cluster, uint32_t chip, const CoreCoord& eth_logical);

}  // namespace tt::tt_metal::streaming_profiler::link_sync
