// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0
//
// NOTE:
// This header is part of the public Metalium API.  Keep it independent from
// generated HAL accessors so that downstream applications can rely on stable
// telemetry data structures.

#pragma once

#include <array>
#include <cstdint>
#include <optional>

namespace tt::tt_fabric {

/**
 * @brief Possible states reported by a fabric router.
 */
enum class FabricTelemetryRouterState : std::uint8_t { Standby = 0, Active = 1, Paused = 2, Draining = 3 };

/**
 * @brief Bitmask of dynamic statistics that may be present in a snapshot.
 */
enum class FabricTelemetryStatistic : std::uint8_t {
    RouterState = 0x01,
    Bandwidth = 0x02,
    HeartbeatTx = 0x04,
    HeartbeatRx = 0x08
};

using FabricTelemetryStatisticMask = std::uint8_t;

/**
 * @brief Compact timestamp representation backed by two 32-bit halves.
 */
struct FabricTelemetryTimestamp {
    std::uint32_t lo = 0;
    std::uint32_t hi = 0;

    [[nodiscard]] constexpr std::uint64_t full() const { return (static_cast<std::uint64_t>(hi) << 32) | lo; }
};

/**
 * @brief Bandwidth counters accumulated by a router direction.
 */
struct FabricTelemetryBandwidthCounters {
    std::uint64_t elapsed_active_cycles = 0;
    std::uint64_t elapsed_cycles = 0;
    std::uint64_t words_sent = 0;
    std::uint64_t packets_sent = 0;
};

/**
 * @brief Per-eRISC dynamic entry.
 */
struct FabricTelemetryEriscEntry {
    FabricTelemetryRouterState router_state = FabricTelemetryRouterState::Standby;
    std::uint64_t tx_heartbeat = 0;
    std::uint64_t rx_heartbeat = 0;
};

/**
 * @brief Dynamic telemetry information that can be sampled from hardware.
 */
struct FabricTelemetryDynamicInfo {
    FabricTelemetryBandwidthCounters tx_bandwidth;
    FabricTelemetryBandwidthCounters rx_bandwidth;
    std::array<FabricTelemetryEriscEntry, 2> erisc{};
};

/**
 * @brief Static information that identifies a router.
 */
struct FabricTelemetryStaticInfo {
    std::uint16_t mesh_id = 0;
    std::uint8_t device_id = 0;
    std::uint8_t direction = 0;
    std::uint32_t fabric_config = 0;
    FabricTelemetryStatisticMask supported_stats = 0;
};

/**
 * @brief Snapshot that contains static information and an optional dynamic payload.
 */
struct FabricTelemetrySnapshot {
    FabricTelemetryStaticInfo static_info;
    std::optional<FabricTelemetryDynamicInfo> dynamic_info;
};

/**
 * @brief Helper to check if a statistic bit is enabled in the provided mask.
 */
[[nodiscard]] constexpr bool telemetry_stat_enabled(FabricTelemetryStatisticMask mask, FabricTelemetryStatistic stat) {
    return (mask & static_cast<FabricTelemetryStatisticMask>(stat)) != 0;
}

}  // namespace tt::tt_fabric
