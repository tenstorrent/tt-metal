// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string_view>
#include <variant>

#include <hostdevcommon/telemetry_sample.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>

// The host side of telemetry samples, which kernels send with TelemetryMetadata (hostdevcommon/telemetry_sample.hpp).
//
// This API is experimental and may change or be removed without notice.
//
//     using namespace tt::tt_metal::experimental::device_observability;
//     using tt::tt_metal::experimental::streaming_profiler::TelemetrySample;
//
//     Callback callback = RegisterCallback(
//         [](const Batch<TelemetrySample>& batch) {
//             for (const TelemetrySample& sample : batch.records<TelemetrySample>()) {
//                 use(sample.metadata().name, sample.metadata().units, sample.core(), sample.payload(),
//                     sample.published_at().time());
//             }
//         },
//         "my-telemetry",
//         /*external_process=*/true);
//
namespace tt::tt_metal::experimental::streaming_profiler {

/** @brief The types a kernel may send a telemetry sample's value as. */
using TelemetryValue = std::variant<uint8_t, uint16_t, uint32_t, uint64_t, float, double>;

/** @brief A telemetry sample: a metric's value, sent with TelemetryMetadata. */
using TelemetrySample = device_observability::Record<TelemetryValue, TelemetryMetadata>;

/** @brief The prefix every telemetry metric's name starts with. */
inline constexpr std::string_view kTelemetryNamePrefix = "tt_";

}  // namespace tt::tt_metal::experimental::streaming_profiler
