// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <hostdevcommon/fabric_common.h>

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"

// The fabric manifest's model. The collector fills in what the router builders know, and the writer
// (write_fabric_manifest) adds what only ControlPlane and the cluster know (routing plane, peer, cross-host,
// wrap, cores) and serializes the manifest.
namespace tt::tt_fabric::manifest {

// A capturable range of L1.
struct L1Region {
    uint32_t address = 0;
    uint32_t size = 0;
    // Set for arrays: size == num_elements * size_per_element.
    std::optional<uint32_t> num_elements;
    std::optional<uint32_t> size_per_element;
    // Schema string, e.g. "u32" or "struct:EDMChannelWorkerLocationInfo" (schema_name()).
    std::string schema;
    // If true, the host zeroes the region before launch (get_fabric_router_addresses_to_clear()).
    bool cleared_by_host = false;
};

// A router's L1 credit counter arrays, shared by every VC that uses counter credits. The to_sender arrays are
// indexed by this router's sender compact index, and the receiver arrays by the peer router's sender compact index
// (the receiver counts credits for the peer's sender channels).
struct L1CreditCounters {
    L1Region to_sender_ack;
    L1Region to_sender_completion;
    L1Region receiver_ack;
    L1Region receiver_completion;
};

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
    // Always reserved, although not always used. Whether a VC uses them is its mesh's credit_transport backing.
    L1CreditCounters credit_counters;
};

// Information about a chip.
struct Chip {
    ZPortRole z_port_role = ZPortRole::NONE;
    std::vector<Router> routers;
};

}  // namespace tt::tt_fabric::manifest
