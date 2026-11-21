// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_umd.hpp"

#include <chrono>
#include <exception>
#include <iostream>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <tt-umd-workload/chip/remote_chip.hpp>
#include <tt-umd-workload/cluster.hpp>
#include <tt-umd/cluster_descriptor.hpp>
#include <tt-umd/soc_descriptor.hpp>
#include <tt-umd/topology/topology_discovery_error.hpp>
#include <tt-umd/topology/topology_discovery_options.hpp>
#include <tt-umd/types/arch.hpp>
#include <tt-umd/types/cluster_descriptor_types.hpp>
#include <tt-umd/types/core_coordinates.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_fabric::capture {

namespace {

// UMD's core for a capture core: an Ethernet core, in translated coordinates.
tt::umd::CoreCoord make_umd_eth_core(CoreXY core) {
    return tt::umd::CoreCoord(core.x, core.y, tt::CoreType::ETH, tt::CoordSystem::TRANSLATED);
}

CoreXY make_core_xy(const tt::umd::CoreCoord& core) {
    return CoreXY{.x = static_cast<uint32_t>(core.x), .y = static_cast<uint32_t>(core.y)};
}

// How many small reads time each candidate channel.
constexpr int NUM_PROBE_READS = 3;

// The gateway's Ethernet channels a remote read can go through: those linked straight to `remote_chip`, else (a chip
// more than one hop away) every active channel of the gateway.
std::set<uint32_t> find_candidate_gateway_channels(
    tt::umd::ClusterDescriptor& descriptor, tt::ChipId gateway_chip, tt::ChipId remote_chip) {
    std::set<uint32_t> channels;
    const auto connections = descriptor.get_ethernet_connections().find(gateway_chip);
    if (connections != descriptor.get_ethernet_connections().end()) {
        for (const auto& [channel, peer] : connections->second) {
            if (std::get<0>(peer) == remote_chip) {
                channels.insert(static_cast<uint32_t>(channel));
            }
        }
    }
    if (channels.empty()) {
        channels = descriptor.get_active_eth_channels(gateway_chip);
    }
    return channels;
}

// Sends `remote_chip`'s reads through the gateway channel that answers small reads fastest. On Wormhole a remote read
// is served by base firmware on the gateway's Ethernet core, which runs only when that core's fabric router hands it
// the core: a router busy with traffic (or stuck spinning on it) rarely does, an idle one often. By default UMD reads
// through the gateway's first active channel, whatever its router is doing. Leaves UMD's default when no channel
// answers.
void route_remote_reads_through_fastest_channel(tt::umd::Cluster& cluster, tt::ChipId remote_chip) {
    tt::umd::ClusterDescriptor& descriptor = *cluster.get_cluster_description();
    const tt::ChipId gateway_chip = descriptor.get_closest_mmio_capable_chip(remote_chip);
    tt::umd::RemoteChip* remote = cluster.get_remote_chip(remote_chip);
    const std::set<uint32_t> default_channels = descriptor.get_active_eth_channels(gateway_chip);
    const tt::umd::CoreCoord probe_core =
        cluster.get_soc_descriptor(remote_chip).get_cores(tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED).front();

    std::optional<uint32_t> fastest_channel;
    std::chrono::nanoseconds fastest_time = std::chrono::nanoseconds::max();
    for (const uint32_t channel : find_candidate_gateway_channels(descriptor, gateway_chip, remote_chip)) {
        remote->set_remote_transfer_ethernet_cores(std::set<uint32_t>{channel});
        try {
            const auto start = std::chrono::steady_clock::now();
            for (int read = 0; read < NUM_PROBE_READS; ++read) {
                uint32_t word = 0;
                cluster.read_from_device(&word, remote_chip, probe_core, 0, sizeof(word));
            }
            const auto elapsed = std::chrono::steady_clock::now() - start;
            std::cerr << "fabric_capture: chip " << remote_chip << " through gateway chip " << gateway_chip
                      << " channel " << channel << ": " << NUM_PROBE_READS << " reads in "
                      << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " us\n";
            if (elapsed < fastest_time) {
                fastest_time = elapsed;
                fastest_channel = channel;
            }
        } catch (const std::exception& error) {
            std::cerr << "fabric_capture: chip " << remote_chip << " through gateway chip " << gateway_chip
                      << " channel " << channel << ": failed: " << error.what() << "\n";
        }
    }

    if (!fastest_channel.has_value()) {
        remote->set_remote_transfer_ethernet_cores(default_channels);
        std::cerr << "fabric_capture: chip " << remote_chip << ": no channel answered; reading through UMD's default\n";
        return;
    }
    remote->set_remote_transfer_ethernet_cores(std::set<uint32_t>{*fastest_channel});
    std::cerr << "fabric_capture: chip " << remote_chip << ": reading through gateway chip " << gateway_chip
              << " channel " << *fastest_channel << "\n";
}

}  // namespace

ClusterFacts collect_umd_cluster_facts(tt::umd::Cluster& cluster) {
    const tt::umd::ClusterDescriptor& descriptor = *cluster.get_cluster_description();
    ClusterFacts facts{.cluster_id = descriptor.get_cluster_id()};
    for (const tt::ChipId chip : cluster.get_target_device_ids()) {
        const tt::umd::SocDescriptor& soc = cluster.get_soc_descriptor(chip);
        FoundChip found{
            .chip = chip,
            .asic_id = descriptor.get_chip_unique_ids().at(chip),
            .arch = descriptor.get_arch(chip),
        };
        found.eth_cores_by_channel.reserve(soc.get_num_eth_channels());
        for (uint32_t channel = 0; channel < soc.get_num_eth_channels(); ++channel) {
            found.eth_cores_by_channel.emplace_back(
                make_core_xy(soc.get_eth_core_for_channel(channel, tt::CoordSystem::TRANSLATED)));
        }

        // Discovery reports a stalled heartbeat at the core it checked, in that core's coordinate system.
        const auto errors = descriptor.get_health_errors().find(chip);
        if (errors != descriptor.get_health_errors().end()) {
            for (const auto& error : errors->second) {
                const auto* heartbeat = std::get_if<tt::umd::error::EthFirmwareHeartbeatError>(&error);
                if (heartbeat == nullptr) {
                    continue;
                }
                facts.skipped_cores.push_back(DiscoverySkippedCore{
                    .asic_id = found.asic_id,
                    .core = make_core_xy(soc.translate_coord_to(heartbeat->data().core, tt::CoordSystem::TRANSLATED)),
                    .reason = heartbeat->message(),
                });
            }
        }
        facts.chips.push_back(std::move(found));
    }
    return facts;
}

UmdDeviceReader::UmdDeviceReader(std::optional<std::string> cluster_id) {
    tt::umd::ClusterOptions options;
    options.num_host_mem_ch_per_mmio_device = 0;
    options.topology_discovery_options.eth_fw_heartbeat_failure = tt::umd::TopologyDiscoveryOptions::Action::IGNORE;
    options.topology_discovery_options.cluster_id = std::move(cluster_id);
    cluster_ = std::make_unique<tt::umd::Cluster>(std::move(options));
    TT_FATAL(
        cluster_->get_cluster_description()->has_authentic_chip_unique_ids(),
        "Fabric capture: UMD didn't read the chips' unique ids from the chips themselves, so capture can't match them "
        "to the manifest's asic_ids");
    for (const tt::ChipId chip : cluster_->get_target_device_ids()) {
        if (cluster_->get_cluster_description()->is_chip_remote(chip)) {
            route_remote_reads_through_fastest_channel(*cluster_, chip);
        }
    }
}

UmdDeviceReader::~UmdDeviceReader() = default;

void UmdDeviceReader::read_l1(int32_t chip, CoreXY core, uint64_t address, std::span<std::byte> out) {
    cluster_->read_from_device(out.data(), chip, make_umd_eth_core(core), address, out.size());
}

uint32_t UmdDeviceReader::read_register(int32_t chip, CoreXY core, uint64_t address) {
    uint32_t value = 0;
    cluster_->read_from_device_reg(&value, chip, make_umd_eth_core(core), address, sizeof(value));
    return value;
}

}  // namespace tt::tt_fabric::capture
