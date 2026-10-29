// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// An external process: a process of its own, apart from the Metal process running the workload, that logs every metric
// the workload's kernels send with kSendToExternalProcess. Registering its callback with external_process set makes it
// the external process, and a Metal process connects to it when it opens its devices.
//
//   metal_example_external_process

#include <chrono>
#include <cstdint>
#include <iostream>
#include <variant>

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <tt-metalium/experimental/device_observability.hpp>
#include <tt_stl/fmt.hpp>

#include "sample.hpp"

namespace device_observability = tt::tt_metal::experimental::device_observability;

// The types a kernel may send a sample's value as.
using Number = std::variant<uint64_t, int64_t, uint32_t, int32_t, double, float>;
using Sample = device_observability::Record<Number, device_observability::SampleMetadata>;

int main() {
    device_observability::Callback callback = device_observability::RegisterCallback(
        [](const device_observability::Batch<Sample>& batch) {
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
                    sample.payload());
            }
        },
        "external-process",
        /*external_process=*/true);

    fmt::print("Listening. Press Enter to stop.\n");
    std::cin.get();
}
