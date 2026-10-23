// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// An external process: a process of its own, apart from the Metal process running the workload, that logs every metric
// the workload's kernels send with kSendToExternalProcess. A process that registers a callback without opening devices
// of its own is the external process, and a Metal process connects to it when it opens its devices.
//
//   metal_example_external_process

#include <chrono>
#include <cstdint>
#include <iostream>
#include <variant>

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <hostdevcommon/sample.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>
#include <tt_stl/fmt.hpp>

namespace streaming_profiler = tt::tt_metal::experimental::streaming_profiler;

// The types a kernel may send a sample's value as.
using Number = std::variant<uint64_t, int64_t, uint32_t, int32_t, double, float>;
using Sample = streaming_profiler::Record<streaming_profiler::Arguments<Number>, streaming_profiler::SampleMetadata>;

int main() {
    streaming_profiler::Callback callback = streaming_profiler::RegisterCallback(
        [](const streaming_profiler::Batch<Sample>& batch) {
            const auto steady_to_system = std::chrono::system_clock::now().time_since_epoch() -
                                          std::chrono::steady_clock::now().time_since_epoch();

            for (const Sample& sample : batch.records<Sample>()) {
                const auto published = std::chrono::floor<std::chrono::microseconds>(
                    sample.published_at().time().time_since_epoch() + steady_to_system);

                fmt::print(
                    "{:%H:%M:%S} chip {} core ({},{}) {}: {}\n",
                    published % std::chrono::days(1),
                    sample.core().chip_id,
                    sample.core().logical.x,
                    sample.core().logical.y,
                    sample.metadata().name,
                    sample.payload()[0]);
            }
        },
        "external-process");

    fmt::print("Listening. Press Enter to stop.\n");
    std::cin.get();
}
