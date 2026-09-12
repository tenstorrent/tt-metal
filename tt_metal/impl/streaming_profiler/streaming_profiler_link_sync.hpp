// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/core_coord.hpp>

namespace tt {
class Cluster;
}

// The device-to-device link sync's contract between the streaming profiler and the fabric router: which eth link of
// a chip pair carries it and which end sends, how often a round goes, and the L1 the two ends own. The device side
// is tools/profiler/sync/eth_ptp_link.hpp; without fabric the profiler runs the ends as resident kernels, with
// fabric the routers on the chosen link run them (fabric_erisc_router.cpp, LINK_SYNC_ROLE).
namespace tt::tt_metal::streaming_profiler::link_sync {

constexpr uint32_t kPaceTicks = 500000;  // a round every 10 ms of the eth tile's 50 MHz refclk
// The top of the active eth core's unreserved region: the pilot's landing place and the frame slots (eth_ptp.hpp
// kPilotOffset, kSlotsOffset), then the stop, done and diagnostic words at kCtlOffset, where the resident kernels
// keep theirs too.
constexpr uint32_t kL1Bytes = 640;
constexpr uint32_t kCtlOffset = 480;

enum class Role : uint32_t { None = 0, Sender = 1, Receiver = 2 };
// TT_METAL_STREAMING_PROFILER_LINK_SYNC=0 leaves the link sync out of a profiler session: no links planned, every
// router's role None. The rest of the profiler runs as usual; it is how the sync's own cost is measured.
bool enabled();
// The control word at the diagnostics' base (eth_ptp_link.hpp kCtlRun/kCtlStop): the sender issues rounds only while
// it reads Run; a resident kernel exits on Stop and then sets the done word behind it.
constexpr uint32_t kCtlRun = 1, kCtlStop = 2;

struct Link {
    uint32_t chip_a = 0, chip_b = 0;  // chip_a < chip_b; chip_a's end sends
    CoreCoord eth_a, eth_b;           // logical eth cores
};

// The one link two connected chips sync over: the lower chip's first eth core connected to the higher, in the
// cluster's order, and the core it connects to; with fabric on, the first such link whose two cores hold routers.
std::optional<Link> link_between(const tt::Cluster& cluster, uint32_t chip_x, uint32_t chip_y);
// What the router on this eth core does for the sync.
Role role_of(const tt::Cluster& cluster, uint32_t chip, const CoreCoord& eth_logical);

}  // namespace tt::tt_metal::streaming_profiler::link_sync
