// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// TelemetryPublish sends a telemetry metric's value to the external process, as a DevicePublish record
// (api/debug/device_publish.h) with TelemetryMetadata (hostdevcommon/telemetry_sample.hpp) and kSendToExternalProcess:
//
//     TelemetryPublish("tt_bytes_processed_total", bytes);
//
// The name is a plain literal that starts with "tt_", and the value is a uint64_t or a double, the types the host's
// TelemetrySample (tt-metalium/experimental/telemetry_sample.hpp) reads: both are checked at compile time, so a call
// site the external process would never receive, a uint32_t value included, fails to build. Cast the value first.
#pragma once

#include <cstdint>
#include <type_traits>

#include "api/debug/device_publish.h"
#include "hostdevcommon/telemetry_sample.hpp"

#define TelemetryPublish(name, value)                                                                           \
    do {                                                                                                        \
        static_assert(                                                                                          \
            sizeof(name) > 3 && (name)[0] == 't' && (name)[1] == 't' && (name)[2] == '_',                       \
            "a telemetry metric's name starts with tt_");                                                       \
        static_assert(                                                                                          \
            std::is_same_v<std::decay_t<decltype(value)>, uint64_t> ||                                          \
                std::is_same_v<std::decay_t<decltype(value)>, double>,                                          \
            "a telemetry value is a uint64_t or a double");                                                     \
        DevicePublish<                                                                                          \
            ::tt::tt_metal::experimental::streaming_profiler::TelemetryMetadata{TT_RECORD_STRING(name)},        \
            ::tt::tt_metal::experimental::device_observability::kSendToExternalProcess>(value);                 \
    } while (0)
