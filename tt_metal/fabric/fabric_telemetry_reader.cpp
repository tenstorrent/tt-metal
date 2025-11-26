// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/fabric_telemetry_reader.hpp>

#include <cstddef>
#include <vector>

#include "tt_metal/fabric/fabric_telemetry_converter.hpp"

#include "tt_metal/api/tt-metalium/experimental/fabric/control_plane.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/llrt/hal.hpp"
#include "tt_metal/llrt/hal/generated/fabric_telemetry.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_stl/assert.hpp"

namespace tt::tt_fabric {

namespace {

struct ChannelContext {
    tt::tt_fabric::chan_id_t channel_id = 0;
};

[[nodiscard]] std::vector<ChannelContext> collect_channel_contexts(
    const tt::tt_fabric::ControlPlane& control_plane, ChipId physical_chip_id) {
    std::vector<ChannelContext> contexts;
    const auto logical_cores = control_plane.get_active_ethernet_cores(physical_chip_id);
    contexts.reserve(logical_cores.size());

    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& chan_map = cluster.get_soc_desc(physical_chip_id).logical_eth_core_to_chan_map;

    for (const auto& logical_core : logical_cores) {
        auto chan_it = chan_map.find(logical_core);
        TT_FATAL(
            chan_it != chan_map.end(),
            "No channel mapping defined for logical ethernet core on chip {}",
            physical_chip_id);
        contexts.push_back(ChannelContext{static_cast<tt::tt_fabric::chan_id_t>(chan_it->second)});
    }

    return contexts;
}

}  // namespace

std::vector<FabricTelemetrySample> read_fabric_telemetry(const tt::tt_fabric::FabricNodeId& fabric_node_id) {
    auto& metal_ctx = tt::tt_metal::MetalContext::instance();
    auto& control_plane = metal_ctx.get_control_plane();
    const auto physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(fabric_node_id);

    const auto& hal = metal_ctx.hal();
    const auto& cluster = metal_ctx.get_cluster();
    const auto& factory = hal.get_fabric_telemetry_factory(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
    const auto telemetry_size = factory.size_of<fabric_telemetry::FabricTelemetry>();
    const auto telemetry_addr = hal.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::FABRIC_TELEMETRY);

    const auto channels = collect_channel_contexts(control_plane, physical_chip_id);
    if (channels.empty()) {
        return {};
    }

    std::vector<FabricTelemetrySample> samples;
    samples.reserve(channels.size());

    std::vector<std::byte> scratch(telemetry_size);
    cluster.l1_barrier(physical_chip_id);

    for (const auto& channel : channels) {
        const auto virtual_core = cluster.get_virtual_eth_core_from_channel(physical_chip_id, channel.channel_id);
        cluster.read_core(scratch.data(), telemetry_size, tt_cxy_pair(physical_chip_id, virtual_core), telemetry_addr);

        const auto view = factory.create_view<fabric_telemetry::FabricTelemetry>(scratch.data());

        auto& sample = samples.emplace_back();
        sample.fabric_node_id = fabric_node_id;
        sample.channel_id = channel.channel_id;
        sample.snapshot = fabric_telemetry_converter::unpack_snapshot_from_hal(view);
    }

    return samples;
}

}  // namespace tt::tt_fabric
