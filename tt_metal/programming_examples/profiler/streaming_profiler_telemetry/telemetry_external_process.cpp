// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// An external process: a process of its own, apart from the Metal process running the workload, that logs every
// telemetry sample the workload's kernels send with TelemetryMetadata and kSendToExternalProcess. Registering its
// callback with external_process set makes it the external process, and a Metal process connects to it when it opens
// its devices.
//
//   metal_example_telemetry_external_process

#include <chrono>
#include <cstdint>
#include <iostream>
#include <variant>

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <tt-metalium/experimental/telemetry_sample.hpp>
#include <tt_stl/fmt.hpp>

namespace device_observability = tt::tt_metal::experimental::device_observability;
namespace streaming_profiler = tt::tt_metal::experimental::streaming_profiler;

int main() {
    device_observability::Callback callback = device_observability::RegisterCallback(
        [](const device_observability::Batch<streaming_profiler::TelemetrySample>& batch) {
            const auto steady_to_system = std::chrono::system_clock::now().time_since_epoch() -
                                          std::chrono::steady_clock::now().time_since_epoch();

            for (const auto& sample : batch.records<streaming_profiler::TelemetrySample>()) {
                const auto published = std::chrono::floor<std::chrono::microseconds>(
                    sample.published_at().time().time_since_epoch() + steady_to_system);

                if (!sample.metadata().name.starts_with(streaming_profiler::kTelemetryNamePrefix)) {
                    fmt::print(
                        "warning: telemetry metric {} doesn't start with {}\n",
                        sample.metadata().name,
                        streaming_profiler::kTelemetryNamePrefix);
                }
                fmt::print(
                    "{:%H:%M:%S} chip {} core ({},{}) {}: {}\n",
                    published % std::chrono::days(1),
                    sample.core().chip_id,
                    sample.core().logical.x,
                    sample.core().logical.y,
                    sample.metadata().name,
                    sample.payload());
            }
        },
        "telemetry-external-process",
        /*external_process=*/true);

    fmt::print("Listening for telemetry. Press Enter to stop.\n");
    std::cin.get();
}
