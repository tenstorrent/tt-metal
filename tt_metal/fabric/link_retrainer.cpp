// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/link_retrainer.hpp"

#include <array>
#include <chrono>
#include <fstream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt_stl/span.hpp>

#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/fabric/physical_system_discovery.hpp"

namespace tt::tt_fabric {

namespace {

using tt::tt_metal::AsicID;
using tt::tt_metal::PhysicalSystemDescriptor;
using tt::tt_metal::experimental::PhysicalNodeId;

// Settle time after a retrain before discovery reads the links, as run_cluster_validation allows.
constexpr auto k_post_retrain_settle_time = std::chrono::seconds(5);

bool any_rank_reports(const tt::tt_metal::distributed::multihost::DistributedContext& ctx, bool local) {
    std::array<std::uint64_t, 1> mine{static_cast<std::uint64_t>(local)};
    std::array<std::uint64_t, 1> any{};
    using ::tt::tt_metal::distributed::multihost::ReduceOp;
    ctx.all_reduce(ttsl::Span<std::uint64_t>(mine), ttsl::Span<std::uint64_t>(any), ReduceOp::MAX);
    return any[0] != 0;
}

// Channels the live descriptor has up, toward any peer.
std::set<CableEnd> live_channels_up(const PhysicalSystemDescriptor& live) {
    std::set<CableEnd> up;
    for (const auto& [host, topology] : live.get_system_graph().asic_connectivity_graph) {
        for (const auto& [asic, edges] : topology) {
            const auto node = live.find_physical_node_id(asic);
            if (!node.has_value()) {
                continue;
            }
            for (const auto& [peer, connections] : edges) {
                for (const auto& connection : connections) {
                    up.insert(CableEnd{*node, connection.src_chan});
                }
            }
        }
    }
    return up;
}

}  // namespace

ClusterLinkRetrainer::ClusterLinkRetrainer(
    tt::Cluster& cluster,
    const tt::tt_metal::Hal& hal,
    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> distributed_context) :
    cluster_(cluster), hal_(hal), distributed_context_(std::move(distributed_context)) {}

bool ClusterLinkRetrainer::can_retrain() const {
    return this->cluster_.arch() == tt::ARCH::BLACKHOLE &&
           this->cluster_.get_target_device_type() == tt::TargetDevice::Silicon &&
           this->cluster_.supports_ethernet_link_retraining();
}

std::vector<EthLinkEndpoint> ClusterLinkRetrainer::retrain(const std::vector<EthLinkEndpoint>& local_endpoints) {
    auto unresponsive =
        retrain_ethernet_links(this->cluster_, this->hal_, *this->distributed_context_, local_endpoints);
    std::this_thread::sleep_for(k_post_retrain_settle_time);
    return unresponsive;
}

PhysicalSystemDescriptor ClusterLinkRetrainer::rediscover() {
    // Discovery reads ethernet connections from the cluster descriptor, which only a refresh rebuilds.
    this->cluster_.rediscover_ethernet_links();
    return tt::tt_metal::run_physical_system_discovery(
        *this->cluster_.get_cluster_desc(), this->distributed_context_, this->cluster_.get_target_device_type());
}

std::vector<MissingCable> retrainable_missing_cables(
    const PhysicalSystemDescriptor& factory, const PhysicalSystemDescriptor& live) {
    const auto delta = tt::tt_metal::experimental::diff_physical_system_descriptors(factory, live);
    if (delta.missing_links.empty()) {
        return {};
    }

    // Missing links are labelled with the factory's ASIC ids, which mean nothing to live: go by address.
    auto node_of = [&](AsicID asic) -> std::optional<PhysicalNodeId> {
        auto node = factory.find_physical_node_id(asic);
        return node.has_value() ? node : live.find_physical_node_id(asic);
    };
    const auto up = live_channels_up(live);

    std::set<MissingCable> cables;
    for (const auto& [src, edges] : delta.missing_links) {
        const auto src_node = node_of(src);
        for (const auto& [dst, connections] : edges) {
            const auto dst_node = node_of(dst);
            if (!src_node.has_value() || !dst_node.has_value() || !live.find_asic_id(*src_node).has_value() ||
                !live.find_asic_id(*dst_node).has_value()) {
                continue;
            }
            for (const auto& connection : connections) {
                CableEnd a{*src_node, connection.src_chan};
                CableEnd b{*dst_node, connection.dst_chan};
                if (up.contains(a) || up.contains(b)) {
                    log_debug(
                        tt::LogFabric,
                        "Not retraining factory cable {} channel {} <-> {} channel {}: a channel is up toward "
                        "another peer",
                        a.node,
                        a.channel,
                        b.node,
                        b.channel);
                    continue;
                }
                if (b < a) {
                    std::swap(a, b);
                }
                cables.insert(MissingCable{a, b});
            }
        }
    }
    return {cables.begin(), cables.end()};
}

std::vector<EthLinkEndpoint> local_endpoints_for_cables(
    const std::vector<MissingCable>& cables, const PhysicalSystemDescriptor& live) {
    std::vector<EthLinkEndpoint> endpoints;
    auto add_if_local = [&](const CableEnd& end, const CableEnd& peer) {
        const auto asic = live.find_asic_id(end.node);
        if (!asic.has_value()) {
            return;
        }
        const auto& descriptor = live.get_asic_descriptors().at(*asic);
        if (descriptor.mpi_rank != live.get_local_mpi_rank()) {
            return;
        }
        endpoints.push_back(EthLinkEndpoint{
            descriptor.umd_unique_id,
            end.channel,
            fmt::format(
                "Factory cable on host {} tray {} location {} channel {}, peer {} channel {}",
                descriptor.host_name,
                *descriptor.tray_id,
                *descriptor.asic_location,
                end.channel,
                peer.node,
                peer.channel)});
    };
    for (const auto& cable : cables) {
        add_if_local(cable.first, cable.second);
        add_if_local(cable.second, cable.first);
    }
    return endpoints;
}

FactoryCableRetrainResult retrain_missing_factory_cables(
    LinkRetrainer& retrainer,
    const tt::tt_metal::distributed::multihost::DistributedContext& distributed_context,
    const PhysicalSystemDescriptor& factory,
    std::unique_ptr<PhysicalSystemDescriptor>& live,
    uint32_t max_rounds) {
    FactoryCableRetrainResult result;
    result.missing_before = retrainable_missing_cables(factory, *live);
    auto missing = result.missing_before;

    // Every rank sees the same global discovery, but the loop is collective, so a disagreement would hang
    // rather than fail: decide on what any rank sees.
    while (result.rounds < max_rounds && any_rank_reports(distributed_context, !missing.empty())) {
        ++result.rounds;
        const auto endpoints = local_endpoints_for_cables(missing, *live);
        log_info(
            tt::LogFabric,
            "Factory link retrain round {}/{}: {} missing cable(s), {} endpoint(s) on this rank",
            result.rounds,
            max_rounds,
            missing.size(),
            endpoints.size());

        const auto unresponsive = retrainer.retrain(endpoints);
        for (const auto& endpoint : unresponsive) {
            log_warning(tt::LogFabric, "Ethernet firmware did not respond to the retrain: {}", endpoint.log_message);
        }

        live = std::make_unique<PhysicalSystemDescriptor>(retrainer.rediscover());
        missing = retrainable_missing_cables(factory, *live);
    }
    result.missing_after = std::move(missing);
    return result;
}

void write_unretrainable_channels(
    const std::filesystem::path& path,
    const std::vector<MissingCable>& cables,
    const PhysicalSystemDescriptor& live,
    uint32_t rounds) {
    struct Entry {
        std::string host;
        uint32_t tray = 0;
        uint32_t location = 0;
        uint32_t channel = 0;
        uint64_t asic_id = 0;
        auto operator<=>(const Entry&) const = default;
    };
    std::set<Entry> entries;
    for (const auto& cable : cables) {
        for (const auto* end : {&cable.first, &cable.second}) {
            const auto asic = live.find_asic_id(end->node);
            if (!asic.has_value()) {
                continue;
            }
            const auto& descriptor = live.get_asic_descriptors().at(*asic);
            entries.insert(
                Entry{descriptor.host_name, *descriptor.tray_id, *descriptor.asic_location, end->channel, **asic});
        }
    }

    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error(fmt::format("cannot open {} for writing", path.string()));
    }
    out << "# Channels that failed to retrain after " << rounds << " attempts.\n";
    out << "# Each entry is one channel endpoint; both ends of an unretrainable cable\n";
    out << "# typically appear as separate entries.\n";
    out << "total_retrain_iterations: " << rounds << "\n";
    out << "unretrainable_channels:\n";
    for (const auto& entry : entries) {
        out << "  - host: " << entry.host << "\n";
        out << "    tray_id: " << entry.tray << "\n";
        out << "    asic_location: " << entry.location << "\n";
        out << "    channel: " << entry.channel << "\n";
        out << fmt::format("    asic_unique_id: 0x{:010x}\n", entry.asic_id);
    }
}

}  // namespace tt::tt_fabric
