// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Retraining the factory cables live discovery did not find, before the control plane places a mesh on them.
//
// A cable that failed to train when the machine came up is often fine after a retrain, and a workload that
// retrains it at startup keeps the routing planes that would otherwise be dropped or, under STRICT system
// health, refused. The retrain itself is the same one run_cluster_validation performs; what is new is
// driving it from the difference between the factory system descriptor and the live descriptor.
//
// Internal header: ControlPlane and MetalEnvImpl are the consumers.

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include <tt-metalium/experimental/fabric/physical_node_id.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>

#include "tt_metal/fabric/ethernet_link_reset.hpp"

namespace tt {
class Cluster;
}  // namespace tt

namespace tt::tt_metal {
class Hal;
}  // namespace tt::tt_metal

namespace tt::tt_metal::distributed::multihost {
class DistributedContext;
}  // namespace tt::tt_metal::distributed::multihost

namespace tt::tt_fabric {

// Retrains ethernet links and rediscovers the system afterwards. Tests substitute their own.
class LinkRetrainer {
public:
    virtual ~LinkRetrainer() = default;

    // Whether this process can retrain. Answered locally: the caller agrees across ranks before acting on it.
    virtual bool can_retrain() const = 0;

    // Collective: every rank calls it, including a rank with no endpoints. Returns the endpoints whose
    // firmware did not respond.
    virtual std::vector<EthLinkEndpoint> retrain(const std::vector<EthLinkEndpoint>& local_endpoints) = 0;

    // Collective. Live discovery of the whole system, run after a retrain.
    virtual tt::tt_metal::PhysicalSystemDescriptor rediscover() = 0;
};

// Retrains through the cluster's ethernet firmware. Blackhole silicon only, with ETH FW >= 1.9.0: the
// Wormhole retrain is not exercised from a workload, and a mock or simulated cluster has no links to train.
class ClusterLinkRetrainer final : public LinkRetrainer {
public:
    ClusterLinkRetrainer(
        tt::Cluster& cluster,
        const tt::tt_metal::Hal& hal,
        std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> distributed_context);

    bool can_retrain() const override;
    std::vector<EthLinkEndpoint> retrain(const std::vector<EthLinkEndpoint>& local_endpoints) override;
    tt::tt_metal::PhysicalSystemDescriptor rediscover() override;

private:
    tt::Cluster& cluster_;
    const tt::tt_metal::Hal& hal_;
    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> distributed_context_;
};

// One end of a cable, by physical address.
struct CableEnd {
    tt::tt_metal::experimental::PhysicalNodeId node;
    uint8_t channel = 0;

    friend bool operator==(const CableEnd&, const CableEnd&) = default;
    friend auto operator<=>(const CableEnd&, const CableEnd&) = default;
};

// A factory cable live discovery did not find. first < second, so a cable has one representation.
struct MissingCable {
    CableEnd first;
    CableEnd second;

    friend bool operator==(const MissingCable&, const MissingCable&) = default;
    friend auto operator<=>(const MissingCable&, const MissingCable&) = default;
};

// The factory cables live is missing that a retrain could bring back, sorted. A cable qualifies when both
// of its chips are live and neither of its channels is up toward some other peer: a channel that trained
// to the wrong peer is miscabled, and retraining it would only take down a working link.
std::vector<MissingCable> retrainable_missing_cables(
    const tt::tt_metal::PhysicalSystemDescriptor& factory, const tt::tt_metal::PhysicalSystemDescriptor& live);

// The ends of `cables` on chips this rank discovered, which are the ones it can retrain.
std::vector<EthLinkEndpoint> local_endpoints_for_cables(
    const std::vector<MissingCable>& cables, const tt::tt_metal::PhysicalSystemDescriptor& live);

struct FactoryCableRetrainResult {
    uint32_t rounds = 0;
    std::vector<MissingCable> missing_before;
    std::vector<MissingCable> missing_after;
};

// Collective. Retrains the cables retrainable_missing_cables reports, for at most max_rounds rounds, replacing
// live with the rediscovered descriptor after each round. Stops early once no rank sees a missing cable.
// max_rounds must be the same on every rank.
FactoryCableRetrainResult retrain_missing_factory_cables(
    LinkRetrainer& retrainer,
    const tt::tt_metal::distributed::multihost::DistributedContext& distributed_context,
    const tt::tt_metal::PhysicalSystemDescriptor& factory,
    std::unique_ptr<tt::tt_metal::PhysicalSystemDescriptor>& live,
    uint32_t max_rounds);

// Writes every end of `cables` to `path` in the format run_cluster_validation uses for
// unretrainable_channels.yaml. Throws on a filesystem error.
void write_unretrainable_channels(
    const std::filesystem::path& path,
    const std::vector<MissingCable>& cables,
    const tt::tt_metal::PhysicalSystemDescriptor& live,
    uint32_t rounds);

}  // namespace tt::tt_fabric
