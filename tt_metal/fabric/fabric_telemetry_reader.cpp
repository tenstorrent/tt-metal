// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/api/tt-metalium/fabric_telemetry_reader.hpp"

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "tt_metal/api/tt-metalium/control_plane.hpp"
#include "tt_metal/fabric/fabric_telemetry_converter.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/llrt/hal.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal {
namespace {

using HalTelemetryStruct = ::tt::tt_fabric::fabric_telemetry::FabricTelemetry;

struct ChannelBinding {
    std::uint8_t channel_id;
    CoreCoord logical_eth_core;
};

std::optional<CoreCoord> find_logical_core_for_channel(const metal_SocDescriptor& soc_desc, std::uint8_t channel_id) {
    for (const auto& [logical_core, mapped_channel] : soc_desc.logical_eth_core_to_chan_map) {
        if (static_cast<std::uint8_t>(mapped_channel) == channel_id) {
            return logical_core;
        }
    }
    return std::nullopt;
}

ChannelBinding make_binding(const CoreCoord& logical_core, std::uint8_t channel_id) {
    return ChannelBinding{
        .channel_id = channel_id,
        .logical_eth_core = logical_core,
    };
}

std::vector<ChannelBinding> get_channel_bindings_for_node(
    const std::unordered_set<CoreCoord>& active_cores, const metal_SocDescriptor& soc_desc) {
    std::vector<ChannelBinding> bindings;
    bindings.reserve(active_cores.size());
    for (const auto& logical_core : active_cores) {
        auto chan_it = soc_desc.logical_eth_core_to_chan_map.find(logical_core);
        if (chan_it == soc_desc.logical_eth_core_to_chan_map.end()) {
            continue;
        }
        bindings.push_back(make_binding(logical_core, static_cast<std::uint8_t>(chan_it->second)));
    }
    return bindings;
}

std::vector<FabricTelemetrySample> read_snapshots_impl(
    const tt::tt_fabric::FabricNodeId& fabric_node_id, std::optional<std::uint8_t> channel_filter) {
    auto& metal_context = MetalContext::instance();
    auto& control_plane = metal_context.get_control_plane();
    auto& cluster = metal_context.get_cluster();
    const auto& hal = metal_context.hal();

    const ChipId physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(fabric_node_id);
    const auto& soc_desc = cluster.get_soc_desc(physical_chip_id);

    const auto& factory = hal.get_fabric_telemetry_factory(HalProgrammableCoreType::ACTIVE_ETH);
    const size_t telemetry_size = factory.size_of<HalTelemetryStruct>();
    const auto l1_addr = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::FABRIC_TELEMETRY);

    auto active_cores = control_plane.get_active_ethernet_cores(physical_chip_id);
    std::vector<std::byte> scratch(telemetry_size);
    std::vector<FabricTelemetrySample> samples;

    auto capture_sample = [&](const ChannelBinding& binding) -> void {
        CoreCoord virtual_eth_core = cluster.get_virtual_eth_core_from_channel(physical_chip_id, binding.channel_id);

        cluster.read_core(
            scratch.data(),
            static_cast<uint32_t>(scratch.size()),
            tt_cxy_pair(physical_chip_id, virtual_eth_core),
            l1_addr);

        auto view = factory.create_view<HalTelemetryStruct>(scratch.data());
        auto snapshot = fabric_telemetry_converter::unpack_snapshot_from_hal(view);

        FabricTelemetrySample sample{
            .channel_id = binding.channel_id,
            .logical_eth_core = binding.logical_eth_core,
            .snapshot = std::move(snapshot),
        };
        samples.emplace_back(std::move(sample));
    };

    if (channel_filter.has_value()) {
        auto logical_core = find_logical_core_for_channel(soc_desc, *channel_filter);
        if (logical_core.has_value() && !active_cores.contains(*logical_core)) {
            logical_core.reset();
        }
        if (!logical_core.has_value()) {
            return samples;
        }
        capture_sample(make_binding(*logical_core, *channel_filter));
        return samples;
    }

    auto bindings = get_channel_bindings_for_node(active_cores, soc_desc);
    samples.reserve(bindings.size());
    for (const auto& binding : bindings) {
        capture_sample(binding);
    }

    return samples;
}

}  // namespace

std::vector<FabricTelemetrySample> ReadFabricTelemetrySnapshots(const tt::tt_fabric::FabricNodeId& fabric_node_id) {
    return read_snapshots_impl(fabric_node_id, std::nullopt);
}

std::optional<FabricTelemetrySample> ReadFabricTelemetrySnapshot(
    const tt::tt_fabric::FabricNodeId& fabric_node_id, std::uint8_t channel_id) {
    auto snapshots = read_snapshots_impl(fabric_node_id, channel_id);
    if (snapshots.empty()) {
        return std::nullopt;
    }
    return std::move(snapshots.front());
}

}  // namespace tt::tt_metal
