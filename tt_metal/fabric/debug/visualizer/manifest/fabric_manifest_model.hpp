// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <hostdevcommon/fabric_common.h>

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"

// The fabric manifest's model. The collector fills in what the router builders know, and the writer
// (write_fabric_manifest) adds what only ControlPlane and the cluster know (routing plane, peer, cross-host,
// wrap, cores) and serializes the manifest.
namespace tt::tt_fabric::manifest {

// High level information about a particular router.
struct RouterIdentity {
    uint32_t eth_chan = 0;
};

// Ethernet link information about a particular router.
struct EthLink {
    eth_chan_directions direction = eth_chan_directions::EAST;
    EdgeCapability edge_capability = EdgeCapability::INTRAMESH_CARDINAL;
    bool is_dispatch_link = false;
};

// Information about the "shape" of a router, i.e. the number of VCs, senders, receivers, and active ERISCs.
struct RouterShape {
    uint32_t num_vcs = 0;
    std::array<uint32_t, builder_config::MAX_NUM_VCS> senders_per_vc = {};
    std::array<uint32_t, builder_config::MAX_NUM_VCS> receivers_per_vc = {};
    uint32_t num_active_eriscs = 0;
    bool channel_trimming_overrides_applied = false;
    bool vc0_bubble_flow_control = false;
};

// Information about a router.
struct Router {
    RouterIdentity identity;
    EthLink link;
    RouterShape shape;
};

// Information about a chip.
struct Chip {
    ZPortRole z_port_role = ZPortRole::NONE;
    std::vector<Router> routers;
};

}  // namespace tt::tt_fabric::manifest
