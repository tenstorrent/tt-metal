// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <hostdevcommon/streaming_profiler_types.hpp>

namespace tt::tt_metal::experimental::streaming_profiler {

/**
 * @brief The unit a telemetry metric's value is expressed in.
 *
 * Never change nor reuse numbers. If something becomes deprecated, mark it as such and do not reuse the number.
 * Reserved numbers may be used in the future.
 */
enum class MetricUnit : uint16_t {
    UNITLESS = 0,
    PERCENT = 1,
    RESERVED_2 = 2,
    RESERVED_3 = 3,
    MEGAHERTZ = 4,
    WATTS = 5,
    MILLIVOLTS = 6,
    VOLTS = 7,
    REVOLUTIONS_PER_MINUTE = 8,
    AMPERES = 9,
    CELSIUS = 10,
    SECONDS = 11,
    MEGABITS_PER_SECOND = 12,
    MEGABYTES = 13,
};

/**
 * @brief The metadata of a telemetry sample, a metric's value, which a kernel sends the external process with
 * TelemetryPublish (api/debug/telemetry_publish.h), with the unit it is in, UNITLESS if none is given:
 *
 *     TelemetryPublish("tt_bytes_processed_total", bytes);
 *     TelemetryPublish("tt_memory_used", megabytes, MetricUnit::MEGABYTES);
 *
 * A telemetry metric's name starts with "tt_", and its value is a uint8_t, uint16_t, uint32_t, uint64_t, float or
 * double: the host reads it as a TelemetrySample (tt-metalium/experimental/telemetry_sample.hpp), which matches only
 * call sites that pass one of those, so a call site that passes any other type, an int32_t included, never reaches it.
 * TelemetryPublish checks both at compile time.
 */
struct TelemetryMetadata {
    device_observability::String name;
    MetricUnit units = MetricUnit::UNITLESS;
};

}  // namespace tt::tt_metal::experimental::streaming_profiler
