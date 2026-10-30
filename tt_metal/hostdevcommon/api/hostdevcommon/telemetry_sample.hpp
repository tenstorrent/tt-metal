// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <hostdevcommon/streaming_profiler_types.hpp>

namespace tt::tt_metal::experimental::streaming_profiler {

/**
 * @brief The metadata of a telemetry sample, a metric's value, which a kernel sends the external process:
 *
 *     DevicePublish<TelemetryMetadata{"tt_bytes_processed_total"_s}, kSendToExternalProcess>(uint64_t{bytes});
 *
 * A telemetry metric's name starts with "tt_", and its value is a uint64_t or a double: the host reads it as a
 * TelemetrySample (tt-metalium/experimental/telemetry_sample.hpp), which matches only call sites that pass one of
 * those two, so a call site that passes any other type, a uint32_t included, never reaches it.
 */
struct TelemetryMetadata {
    device_observability::String name;
};

}  // namespace tt::tt_metal::experimental::streaming_profiler
