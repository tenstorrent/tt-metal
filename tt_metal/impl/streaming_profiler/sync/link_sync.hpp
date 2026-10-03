// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device_types.hpp>
#include "hostdev/streaming_profiler_common.h"
#include "impl/context/context_types.hpp"

namespace tt::tt_fabric {
class FabricNodeId;
}

namespace tt::tt_metal {

class Device;
class Hal;
class MetalContext;

namespace streaming_profiler {

// Without fabric the profiler runs the ends as resident kernels; with fabric the routers on the chosen links run them
// under LINK_SYNC_ROLE in fabric_erisc_router.cpp.
namespace link_sync {

struct Link {
    uint32_t chip_a = 0, chip_b = 0;  // chip_a < chip_b; chip_a's end sends
    CoreCoord eth_a, eth_b;
};
// Every connected eth pair between the two chips or, with fabric on, only those on the fabric's active channels.
std::vector<Link> links_between(MetalContext& mc, uint32_t chip_x, uint32_t chip_y);

// The role of this eth core's end on the link the profiler syncs through it, or None if it syncs none.
kernel_profiler::LinkSyncRole role_of(MetalContext& mc, ChipId chip, const CoreCoord& eth_logical);
// The link end's kernel_profiler::LinkSyncL1 on every active eth core: the top of ACTIVE_ETH UNRESERVED, the same place
// whether a resident kernel or a router runs the end.
uint32_t l1_addr(const Hal& hal);
// The top of the L1 a fabric router may load into, given the router's own limit: below the link end's LinkSyncL1
// whenever the profiler captures.
uint32_t router_l1_limit(const MetalContext& mc, uint32_t router_limit);
// The named compile-time args fabric_erisc_router.cpp reads whenever the JIT defines PROFILE_STREAMING on Blackhole:
// LINK_SYNC_ROLE, the role of the link end the router runs (only ERISC0's router runs one, and only where the profiler
// captures), and LINK_SYNC_L1_ADDR.
void add_router_compile_args(
    MetalContext& mc,
    uint32_t risc_id,
    const tt_fabric::FabricNodeId& node,
    const CoreCoord& eth_logical,
    std::unordered_map<std::string, uint32_t>& named_args);
// Zeroes every active eth core's link-end slots, control word and done word. It runs before the fabric routers start,
// so a router end never reads what an earlier process left there.
void zero_link_end_l1(std::span<Device* const> devices, ContextId context_id);

}  // namespace link_sync

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
