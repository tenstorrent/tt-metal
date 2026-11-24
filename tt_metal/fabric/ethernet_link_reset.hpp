// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Retraining ethernet links through the ethernet firmware: the RETRAIN_FORCE flag on Wormhole, the base
// firmware mailbox on Blackhole.
//
// Every call here acts on endpoints this process can reach. A cable has two endpoints, and when they sit on
// different hosts each host resets its own end, so the cross-host sequencing lives in the collective
// retrain_ethernet_links rather than in the callers.
//
// Internal header: shared by the control plane and the scaleout validation tooling, not installed.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <tt-umd/types/cluster_descriptor_types.hpp>

namespace tt {
class Cluster;
}

namespace tt::tt_metal {
class Hal;
}

namespace tt::tt_metal::distributed::multihost {
class DistributedContext;
}

namespace tt::tt_fabric {

// One end of an ethernet link, addressed by the local chip and its ethernet channel.
struct EthLinkEndpoint {
    ChipId chip_id;
    uint32_t channel;
    // Names the endpoint in the reset logs, e.g. by host, tray and ASIC location.
    std::string log_message;
};

// Stops any Metal kernel left on the endpoints' ERISC0s (e.g. fabric routers from a killed workload) and
// returns the cores to base firmware. Blackhole only.
void return_links_to_base_firmware(
    const tt::Cluster& cluster, const tt::tt_metal::Hal& hal, const std::vector<EthLinkEndpoint>& endpoints);

// Brings the endpoints' ports down. Blackhole only. Returns the endpoints whose base firmware did not process
// the message in time.
std::vector<EthLinkEndpoint> send_port_down_msg_to_links(
    const tt::Cluster& cluster, const tt::tt_metal::Hal& hal, const std::vector<EthLinkEndpoint>& endpoints);

// Retrains the endpoints. Returns the endpoints whose firmware did not process a retrain message in time; an
// endpoint missing from the result was retrained, which says nothing about whether its link came up.
//
// Collective: every rank in distributed_context must call this, including ranks with no endpoints. On
// Blackhole every rank brings its ports down before any rank reinitializes, so both ends of a cross-host
// cable are down when either starts training.
std::vector<EthLinkEndpoint> retrain_ethernet_links(
    const tt::Cluster& cluster,
    const tt::tt_metal::Hal& hal,
    const tt::tt_metal::distributed::multihost::DistributedContext& distributed_context,
    const std::vector<EthLinkEndpoint>& endpoints);

}  // namespace tt::tt_fabric
