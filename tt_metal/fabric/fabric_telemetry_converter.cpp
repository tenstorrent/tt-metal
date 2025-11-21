// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/fabric_telemetry_converter.hpp"

#include <algorithm>

namespace tt::tt_metal::fabric_telemetry_converter {
namespace {

constexpr std::uint32_t lower_32_bits(std::uint64_t value) { return static_cast<std::uint32_t>(value & 0xFFFFFFFFull); }

constexpr std::uint32_t upper_32_bits(std::uint64_t value) {
    return static_cast<std::uint32_t>((value >> 32) & 0xFFFFFFFFull);
}

void pack_timestamp(std::uint64_t value, ::tt::tt_fabric::fabric_telemetry::RiscTimestampV2::View dst) {
    dst.full() = value;
    dst.lo() = lower_32_bits(value);
    dst.hi() = upper_32_bits(value);
}

std::uint64_t unpack_timestamp(::tt::tt_fabric::fabric_telemetry::RiscTimestampV2::ConstView src) {
    return (static_cast<std::uint64_t>(src.hi()) << 32) | static_cast<std::uint64_t>(src.lo());
}

void pack_bandwidth(
    const tt::tt_fabric::FabricTelemetryBandwidthCounters& src,
    ::tt::tt_fabric::fabric_telemetry::BandwidthTelemetry::View dst) {
    pack_timestamp(src.elapsed_active_cycles, dst.elapsed_active_cycles());
    pack_timestamp(src.elapsed_cycles, dst.elapsed_cycles());
    dst.num_words_sent() = src.words_sent;
    dst.num_packets_sent() = src.packets_sent;
}

tt::tt_fabric::FabricTelemetryBandwidthCounters unpack_bandwidth(
    ::tt::tt_fabric::fabric_telemetry::BandwidthTelemetry::ConstView src) {
    tt::tt_fabric::FabricTelemetryBandwidthCounters dst{};
    dst.elapsed_active_cycles = unpack_timestamp(src.elapsed_active_cycles());
    dst.elapsed_cycles = unpack_timestamp(src.elapsed_cycles());
    dst.words_sent = src.num_words_sent();
    dst.packets_sent = src.num_packets_sent();
    return dst;
}

constexpr ::tt::tt_fabric::fabric_telemetry::RouterState to_hal_router_state(
    tt::tt_fabric::FabricTelemetryRouterState state) {
    return static_cast<::tt::tt_fabric::fabric_telemetry::RouterState>(state);
}

constexpr tt::tt_fabric::FabricTelemetryRouterState from_hal_router_state(
    ::tt::tt_fabric::fabric_telemetry::RouterState state) {
    return static_cast<tt::tt_fabric::FabricTelemetryRouterState>(state);
}

void pack_erisc_entry(
    const tt::tt_fabric::FabricTelemetryEriscEntry& src,
    ::tt::tt_fabric::fabric_telemetry::EriscDynamicEntry::View dst) {
    dst.router_state() = to_hal_router_state(src.router_state);
    pack_timestamp(src.tx_heartbeat, dst.tx_heartbeat());
    pack_timestamp(src.rx_heartbeat, dst.rx_heartbeat());
}

tt::tt_fabric::FabricTelemetryEriscEntry unpack_erisc_entry(
    ::tt::tt_fabric::fabric_telemetry::EriscDynamicEntry::ConstView src) {
    tt::tt_fabric::FabricTelemetryEriscEntry dst{};
    dst.router_state = from_hal_router_state(src.router_state());
    dst.tx_heartbeat = unpack_timestamp(src.tx_heartbeat());
    dst.rx_heartbeat = unpack_timestamp(src.rx_heartbeat());
    return dst;
}

void zero_dynamic_info(::tt::tt_fabric::fabric_telemetry::DynamicInfo::View dst) {
    tt::tt_fabric::FabricTelemetryDynamicInfo zero{};
    pack_bandwidth(zero.tx_bandwidth, dst.tx_bandwidth());
    pack_bandwidth(zero.rx_bandwidth, dst.rx_bandwidth());
    static constexpr tt::tt_fabric::FabricTelemetryEriscEntry kEmpty{};
    auto erisc_dst = dst.erisc();
    for (size_t i = 0; i < erisc_dst.size(); ++i) {
        pack_erisc_entry(kEmpty, erisc_dst[i]);
    }
}

}  // namespace

void pack_static_info_to_hal(
    const tt::tt_fabric::FabricTelemetryStaticInfo& src, ::tt::tt_fabric::fabric_telemetry::StaticInfo::View dst) {
    dst.mesh_id() = src.mesh_id;
    dst.device_id() = src.device_id;
    dst.direction() = src.direction;
    dst.fabric_config() = src.fabric_config;
    dst.supported_stats() = static_cast<::tt::tt_fabric::fabric_telemetry::DynamicStatistics>(src.supported_stats);
}

void pack_dynamic_info_to_hal(
    const tt::tt_fabric::FabricTelemetryDynamicInfo& src, ::tt::tt_fabric::fabric_telemetry::DynamicInfo::View dst) {
    pack_bandwidth(src.tx_bandwidth, dst.tx_bandwidth());
    pack_bandwidth(src.rx_bandwidth, dst.rx_bandwidth());

    auto erisc_dst = dst.erisc();
    const auto count = std::min(erisc_dst.size(), src.erisc.size());
    for (size_t i = 0; i < count; ++i) {
        pack_erisc_entry(src.erisc[i], erisc_dst[i]);
    }
    for (size_t i = count; i < erisc_dst.size(); ++i) {
        static constexpr tt::tt_fabric::FabricTelemetryEriscEntry kEmpty{};
        pack_erisc_entry(kEmpty, erisc_dst[i]);
    }
}

void pack_snapshot_to_hal(
    const tt::tt_fabric::FabricTelemetrySnapshot& src, ::tt::tt_fabric::fabric_telemetry::FabricTelemetry::View dst) {
    pack_static_info_to_hal(src.static_info, dst.static_info());
    if (src.dynamic_info.has_value()) {
        pack_dynamic_info_to_hal(*src.dynamic_info, dst.dynamic_info());
    } else {
        zero_dynamic_info(dst.dynamic_info());
    }
}

void pack_snapshot_to_hal(
    const tt::tt_fabric::FabricTelemetrySnapshot& src,
    ::tt::tt_fabric::fabric_telemetry::FabricTelemetryStaticOnly::View dst) {
    pack_static_info_to_hal(src.static_info, dst.static_info());
}

tt::tt_fabric::FabricTelemetryStaticInfo unpack_static_info_from_hal(
    ::tt::tt_fabric::fabric_telemetry::StaticInfo::ConstView src) {
    tt::tt_fabric::FabricTelemetryStaticInfo dst{};
    dst.mesh_id = src.mesh_id();
    dst.device_id = src.device_id();
    dst.direction = src.direction();
    dst.fabric_config = src.fabric_config();
    dst.supported_stats = static_cast<tt::tt_fabric::FabricTelemetryStatisticMask>(src.supported_stats());
    return dst;
}

tt::tt_fabric::FabricTelemetryDynamicInfo unpack_dynamic_info_from_hal(
    ::tt::tt_fabric::fabric_telemetry::DynamicInfo::ConstView src) {
    tt::tt_fabric::FabricTelemetryDynamicInfo dst{};
    dst.tx_bandwidth = unpack_bandwidth(src.tx_bandwidth());
    dst.rx_bandwidth = unpack_bandwidth(src.rx_bandwidth());

    auto erisc_src = src.erisc();
    const auto count = std::min(erisc_src.size(), dst.erisc.size());
    for (size_t i = 0; i < count; ++i) {
        dst.erisc[i] = unpack_erisc_entry(erisc_src[i]);
    }
    for (size_t i = count; i < dst.erisc.size(); ++i) {
        dst.erisc[i] = tt::tt_fabric::FabricTelemetryEriscEntry{};
    }
    return dst;
}

tt::tt_fabric::FabricTelemetrySnapshot unpack_snapshot_from_hal(
    ::tt::tt_fabric::fabric_telemetry::FabricTelemetry::ConstView src) {
    tt::tt_fabric::FabricTelemetrySnapshot snapshot{};
    snapshot.static_info = unpack_static_info_from_hal(src.static_info());
    if (snapshot.static_info.supported_stats != 0) {
        snapshot.dynamic_info = unpack_dynamic_info_from_hal(src.dynamic_info());
    }
    return snapshot;
}

tt::tt_fabric::FabricTelemetrySnapshot unpack_snapshot_from_hal(
    ::tt::tt_fabric::fabric_telemetry::FabricTelemetryStaticOnly::ConstView src) {
    tt::tt_fabric::FabricTelemetrySnapshot snapshot{};
    snapshot.static_info = unpack_static_info_from_hal(src.static_info());
    return snapshot;
}

}  // namespace tt::tt_metal::fabric_telemetry_converter
