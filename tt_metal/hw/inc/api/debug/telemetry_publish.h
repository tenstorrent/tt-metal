// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// TelemetryPublish sends a telemetry metric's value to the external process, as a DevicePublish record
// (api/debug/device_publish.h) with TelemetryMetadata (hostdevcommon/telemetry_sample.hpp) and kSendToExternalProcess:
//
//     TelemetryPublish("tt_bytes_processed_total", bytes);
//     TelemetryPublish("tt_temperature", celsius, MetricUnit::CELSIUS);
//
// The third argument, the MetricUnit the value is in, is optional and defaults to MetricUnit::UNITLESS. MetricUnit
// names the enum without its namespace inside the macro.
//
// The name is a plain literal that starts with "tt_", and the value is a uint8_t, uint16_t, uint32_t, uint64_t, float
// or double, the types the host's TelemetrySample (tt-metalium/experimental/telemetry_sample.hpp) reads: both are
// checked at compile time, so a call site the external process would never receive, an int32_t value included, fails
// to build. Cast the value first.
#pragma once

#include <cstdint>
#include <type_traits>

#include "api/debug/device_publish.h"
#include "hostdevcommon/telemetry_sample.hpp"

#define TelemetryPublish(...)                                                                    \
    TT_TELEMETRY_PUBLISH_SELECT(__VA_ARGS__, TT_TELEMETRY_PUBLISH_WITH_UNITS, TT_TELEMETRY_PUBLISH_UNITLESS, ) \
    (__VA_ARGS__)

#define TT_TELEMETRY_PUBLISH_SELECT(_1, _2, _3, macro, ...) macro
#define TT_TELEMETRY_PUBLISH_UNITLESS(name, value) TT_TELEMETRY_PUBLISH_WITH_UNITS(name, value, MetricUnit::UNITLESS)
#define TT_TELEMETRY_PUBLISH_WITH_UNITS(name, value, units)                                                     \
    do {                                                                                                        \
        using ::tt::tt_metal::experimental::streaming_profiler::MetricUnit;                                     \
        static_assert(                                                                                          \
            sizeof(name) > 3 && (name)[0] == 't' && (name)[1] == 't' && (name)[2] == '_',                       \
            "a telemetry metric's name starts with tt_");                                                       \
        static_assert(                                                                                          \
            std::is_same_v<std::decay_t<decltype(value)>, uint8_t> ||                                           \
                std::is_same_v<std::decay_t<decltype(value)>, uint16_t> ||                                      \
                std::is_same_v<std::decay_t<decltype(value)>, uint32_t> ||                                      \
                std::is_same_v<std::decay_t<decltype(value)>, uint64_t> ||                                      \
                std::is_same_v<std::decay_t<decltype(value)>, float> ||                                         \
                std::is_same_v<std::decay_t<decltype(value)>, double>,                                          \
            "a telemetry value is a uint8_t, uint16_t, uint32_t, uint64_t, float or double");                   \
        DevicePublish<                                                                                          \
            ::tt::tt_metal::experimental::streaming_profiler::TelemetryMetadata{TT_RECORD_STRING(name), units}, \
            ::tt::tt_metal::experimental::device_observability::kSendToExternalProcess>(value);                 \
    } while (0)
