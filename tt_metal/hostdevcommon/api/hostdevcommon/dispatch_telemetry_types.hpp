// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tt::tt_metal {
namespace detail {

constexpr uint32_t pack(const char (&s)[5]) {
    return (uint32_t(s[0]) << 24) | (uint32_t(s[1]) << 16) | (uint32_t(s[2]) << 8) | (uint32_t(s[3]));
}

}  // namespace detail

// Increment only when breaking changes occur
constexpr uint32_t DISPATCH_TELEMETRY_VERSION = 1;

/**
 * @brief Expected signature for validating that a telemetry buffer contains dispatch telemetry data.
 */
constexpr uint32_t DISPATCH_CORE_TELEMETRY_SIGNATURE = detail::pack("DISP");
constexpr uint32_t PREFETCH_CORE_TELEMETRY_SIGNATURE = detail::pack("PREF");

// Used to invalidate the telemetry buffer
constexpr uint32_t INVALID_TELEMETRY_SIGNATURE = 0;

struct __attribute__((packed, aligned(4))) PrefetchCoreTelemetry {
    uint32_t version = DISPATCH_TELEMETRY_VERSION;
    uint32_t signature = PREFETCH_CORE_TELEMETRY_SIGNATURE;
    uint32_t upstream_blocked_count = 0;
    uint32_t upstream_unblocked_count = 0;
    uint32_t command_count = 0;
};

struct __attribute__((packed, aligned(4))) DispatchCoreTelemetry {
    uint32_t version = DISPATCH_TELEMETRY_VERSION;
    uint32_t signature = DISPATCH_CORE_TELEMETRY_SIGNATURE;
    uint32_t upstream_blocked_count = 0;
    uint32_t upstream_unblocked_count = 0;
    uint32_t program_count = 0;
};

// Used to determine the size of the L1 buffer that dispatch_mem_map allocates
// Note: If new telemetry types are added, update this calculation
constexpr size_t DISPATCH_TELEMETRY_SIZE = std::max(sizeof(DispatchCoreTelemetry), sizeof(PrefetchCoreTelemetry));
}  // namespace tt::tt_metal
