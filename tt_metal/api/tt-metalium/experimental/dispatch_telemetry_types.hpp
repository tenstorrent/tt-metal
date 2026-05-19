// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
namespace {
constexpr uint32_t pack(const char (&s)[5]) {
    return (uint32_t(s[0]) << 24) | (uint32_t(s[1]) << 16) | (uint32_t(s[2]) << 8) | (uint32_t(s[3]));
}
}  // namespace
namespace tt::tt_metal {
/**
 * @brief Expected signature for validating that a telemetry buffer contains dispatch telemetry data.
 */
constexpr uint32_t DISPATCH_TELEMETRY_SIGNATURE = pack("DISP");
constexpr uint32_t DISPATCH_TELEMETRY_VERSION = 1;

constexpr uint32_t PREFETCH_TELEMETRY_SIGNATURE = pack("PREF");
constexpr uint32_t PREFETCH_TELEMETRY_VERSION = 1;

// Used to invalidate the telemetry buffer
constexpr uint32_t INVALID_TELEMETRY_SIGNATURE = 0;

/**
 * @brief Telemetry for prefetch.
 */
struct PrefetchTelemetry {
    uint32_t version = PREFETCH_TELEMETRY_VERSION;
    uint32_t signature = PREFETCH_TELEMETRY_SIGNATURE;
    uint64_t blocked_by_host_count = 0;
    uint64_t unblocked_by_host_count = 0;
    uint64_t command_count = 0;
};

/**
 * @brief Telemetry for dispatch.
 */
struct DispatchTelemetry {
    uint32_t version = DISPATCH_TELEMETRY_VERSION;
    uint32_t signature = DISPATCH_TELEMETRY_SIGNATURE;
    uint64_t blocked_by_host_count = 0;
    uint64_t unblocked_by_host_count = 0;
};

// Used to determine the size of the L1 buffer that dispatch_mem_map allocates
// Note: If new telemetry types are added, update this calculation
constexpr size_t DISPATCH_TELEMETRY_SIZE = std::max(sizeof(DispatchTelemetry), sizeof(PrefetchTelemetry));
}  // namespace tt::tt_metal
