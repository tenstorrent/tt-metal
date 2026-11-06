// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_umd.hpp"

#include <utility>
#include <variant>
#include <vector>

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

}  // namespace

ClusterFacts collect_umd_cluster_facts(tt::umd::Cluster& cluster) {
    const tt::umd::ClusterDescriptor& descriptor = *cluster.get_cluster_description();
    ClusterFacts facts{.cluster_id = descriptor.get_cluster_id()};
    for (const tt::ChipId chip : cluster.get_target_device_ids()) {
        const tt::umd::SocDescriptor& soc = cluster.get_soc_descriptor(chip);
        FoundChip found{
            .chip = chip,
            .asic_id = descriptor.get_chip_unique_ids().at(chip),
            .arch = tt::arch_to_str(descriptor.get_arch(chip)),
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
