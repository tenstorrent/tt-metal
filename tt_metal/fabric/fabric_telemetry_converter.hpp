// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <tt-metalium/experimental/fabric/fabric_telemetry.hpp>

namespace tt::tt_fabric::fabric_telemetry_converter {
namespace detail {

template <typename TimestampView>
std::uint64_t unpack_timestamp(const TimestampView& src) {
    return (static_cast<std::uint64_t>(src.hi()) << 32) | static_cast<std::uint64_t>(src.lo());
}

template <typename BandwidthView>
tt::tt_fabric::FabricTelemetryBandwidthCounters unpack_bandwidth(const BandwidthView& src) {
    tt::tt_fabric::FabricTelemetryBandwidthCounters dst{};
    dst.elapsed_active_cycles = unpack_timestamp(src.elapsed_active_cycles());
    dst.elapsed_cycles = unpack_timestamp(src.elapsed_cycles());
    dst.words_sent = src.num_words_sent();
    dst.packets_sent = src.num_packets_sent();
    return dst;
}

template <typename EriscView>
tt::tt_fabric::FabricTelemetryEriscEntry unpack_erisc_entry(const EriscView& src) {
    tt::tt_fabric::FabricTelemetryEriscEntry dst{};
    dst.router_state = static_cast<tt::tt_fabric::FabricTelemetryRouterState>(src.router_state());
    dst.tx_heartbeat = unpack_timestamp(src.tx_heartbeat());
    dst.rx_heartbeat = unpack_timestamp(src.rx_heartbeat());
    return dst;
}

}  // namespace detail

template <typename StaticInfoConstView>
tt::tt_fabric::FabricTelemetryStaticInfo unpack_static_info_from_hal(const StaticInfoConstView& src) {
    tt::tt_fabric::FabricTelemetryStaticInfo dst{};
    dst.mesh_id = src.mesh_id();
    dst.device_id = src.device_id();
    dst.direction = src.direction();
    dst.fabric_config = src.fabric_config();
    dst.supported_stats = static_cast<tt::tt_fabric::FabricTelemetryStatisticMask>(src.supported_stats());
    return dst;
}

template <typename DynamicInfoConstView>
tt::tt_fabric::FabricTelemetryDynamicInfo unpack_dynamic_info_from_hal(const DynamicInfoConstView& src) {
    tt::tt_fabric::FabricTelemetryDynamicInfo dst{};
    dst.tx_bandwidth = detail::unpack_bandwidth(src.tx_bandwidth());
    dst.rx_bandwidth = detail::unpack_bandwidth(src.rx_bandwidth());

    auto erisc_src = src.erisc();
    const auto count = std::min(erisc_src.size(), dst.erisc.size());
    for (size_t i = 0; i < count; ++i) {
        dst.erisc[i] = detail::unpack_erisc_entry(erisc_src[i]);
    }
    for (size_t i = count; i < dst.erisc.size(); ++i) {
        dst.erisc[i] = tt::tt_fabric::FabricTelemetryEriscEntry{};
    }
    return dst;
}

template <typename FabricTelemetryView>
tt::tt_fabric::FabricTelemetrySnapshot unpack_snapshot_from_hal(const FabricTelemetryView& src) {
    tt::tt_fabric::FabricTelemetrySnapshot snapshot{};
    snapshot.static_info = unpack_static_info_from_hal(src.static_info());
    if constexpr (requires { src.dynamic_info(); }) {
        if (snapshot.static_info.supported_stats != 0) {
            snapshot.dynamic_info = unpack_dynamic_info_from_hal(src.dynamic_info());
        }
    }
    return snapshot;
}

}  // namespace tt::tt_fabric::fabric_telemetry_converter
