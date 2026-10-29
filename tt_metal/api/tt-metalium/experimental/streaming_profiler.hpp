// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/experimental/device_observability.hpp>
#include <hostdevcommon/streaming_profiler_types.hpp>

// The streaming profiler's records. Its markers (DeviceZoneScopedN, DeviceTimestampedData, DeviceRecordEvent) publish
// them over device observability, whose RegisterCallback delivers them like any other record.
//
// This API is experimental and may change or be removed without notice.
//
//     using namespace tt::tt_metal::experimental::device_observability;
//     using tt::tt_metal::experimental::streaming_profiler::Zone;
//
//     Callback callback = RegisterCallback(
//         [](const Batch<Zone>& batch) {
//             for (const Zone& zone : batch.records<Zone>()) {
//                 use(zone.metadata().name, zone.core().logical, zone.payload().duration());
//             }
//         },
//         "my-tool");
//
namespace tt::tt_metal::experimental::streaming_profiler {

/** @brief One closed DeviceZoneScopedN scope. */
using Zone = device_observability::Record<device_observability::Interval, ZoneMetadata>;
/** @brief One DeviceTimestampedData marker and the values it recorded. */
using TimestampedData = device_observability::Record<device_observability::Variadic<uint64_t>, TimestampedDataMetadata>;
/** @brief One DeviceRecordEvent marker. */
using Event = device_observability::Record<void, EventMetadata>;

}  // namespace tt::tt_metal::experimental::streaming_profiler
