// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "hostdev/streaming_profiler_common.h"

namespace tt {
class Cluster;
}
namespace tt::tt_fabric {
class ControlPlane;
class FabricContext;
class FabricNodeId;
}  // namespace tt::tt_fabric

namespace tt::tt_metal {

class Hal;

// The link sync measures the offset and rate between two chips' refclks over an eth link between them, as part of the
// clock sync between chips. Its ports are the active eth cores on the two sides of the link. They exchange timestamped
// frames, and the host fits a line to their stamps. Without fabric, each port runs as a resident kernel on its eth
// core. With fabric, that core's router runs it from its main loop instead.
namespace streaming_profiler::link_sync {

struct Link {
    CoreCoord eth_a, eth_b;  // eth_a is on the lower chip id, whose port is the transmitter
};
// Returns every connected eth pair between the two chips. With fabric on (`control_plane` non-null), returns only the
// pairs on its active channels.
std::vector<Link> links_between(
    const tt::Cluster& cluster, const tt_fabric::ControlPlane* control_plane, uint32_t chip_x, uint32_t chip_y);

// Returns the address of a port's LinkSyncL1, which sits at the top of every active eth core's unreserved L1.
uint32_t l1_addr(const Hal& hal);
// Returns the end of the L1 a fabric router may use. When this process profiles its mesh devices, that is the start of
// the port's LinkSyncL1, which takes the top 384 B of the eth core's unreserved L1. Otherwise this returns
// `router_limit`.
uint32_t router_l1_limit(const tt_fabric::FabricContext& fabric, uint32_t router_limit);
// Returns the named compile-time args that the link sync kernel reads. LINK_SYNC_ROLE is the role of the port,
// LINK_SYNC_L1_ADDR is the address of its LinkSyncL1, and LINK_SYNC_CHECK is whether the sync check runs.
std::unordered_map<std::string, uint32_t> compile_args(
    kernel_profiler::LinkSyncRole role, uint32_t link_l1, bool sync_check);
// Adds the link-sync compile-time args (LINK_SYNC_ROLE, LINK_SYNC_L1_ADDR, LINK_SYNC_CHECK) to a Blackhole fabric
// router's kernel when the streaming profiler is on, which is when the router compiles its clock sync port. Only
// ERISC0's router on a link the profiler syncs gets a transmitter or receiver role. Every other router gets None, and
// its hook compiles to nothing.
void add_router_compile_args(
    const tt_fabric::FabricContext& fabric,
    uint32_t risc_id,
    const tt_fabric::FabricNodeId& node,
    const CoreCoord& eth_logical,
    std::unordered_map<std::string, uint32_t>& named_args);
// Zeroes a port's frame slots and its control and done words on the eth core at `virt`.
void zero_port(const Hal& hal, tt::Cluster& cluster, uint32_t chip, const CoreCoord& virt);

}  // namespace streaming_profiler::link_sync
}  // namespace tt::tt_metal
