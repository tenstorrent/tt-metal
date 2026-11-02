// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The streaming profiler's record metadata, for its markers on device observability (device_observability_types.hpp).
// Compiled as C++17 for the device and C++20 for the host.
#pragma once

#include "hostdevcommon/device_observability_types.hpp"

namespace tt::tt_metal::experimental::streaming_profiler {

/** @brief The unit a consumer should present a zone's duration in. The device and the host still count in cycles. */
enum class TimeUnit : uint8_t { Cycles, Nanoseconds, Microseconds, Milliseconds };

/** @brief A zone's metadata: DeviceZoneScopedN sets its name, DeviceZoneScopedNC its color too, and DeviceZoneScopedNCU
 * its unit too. */
struct ZoneMetadata {
    device_observability::String name;
    uint32_t color = 0;  // 0xRRGGBB, or 0 for its processor's color
    TimeUnit unit = TimeUnit::Cycles;
};
/** @brief A DeviceRecordEvent marker's metadata. */
struct EventMetadata {
    device_observability::String name;
};
/** @brief A DeviceTimestampedData marker's metadata. */
struct TimestampedDataMetadata {
    device_observability::String name;
};

}  // namespace tt::tt_metal::experimental::streaming_profiler
