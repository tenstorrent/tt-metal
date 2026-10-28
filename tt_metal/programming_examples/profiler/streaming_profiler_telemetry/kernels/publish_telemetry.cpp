// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Sends three telemetry metrics to the external process once per interval: tt_loop_iteration, the iteration it's on,
// tt_bytes_processed_total, the bytes this core has handled so far, and tt_progress, the fraction of its samples it
// has sent. The bytes are made up: each sample adds this core's runtime argument. Telemetry values are uint64_t or
// double, so each call passes one of those.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/device_publish.h"
#include "hostdevcommon/telemetry_sample.hpp"

void kernel_main() {
    using namespace tt::tt_metal::experimental::streaming_profiler;
    const uint32_t samples = get_arg_val<uint32_t>(0);
    const uint32_t interval_cycles = get_arg_val<uint32_t>(1);
    const uint32_t bytes_per_sample = get_arg_val<uint32_t>(2);
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < samples; i++) {
        riscv_wait(interval_cycles);
        bytes += bytes_per_sample;
        const double progress = static_cast<double>(i + 1) / static_cast<double>(samples);
        DevicePublish<TelemetryMetadata{"tt_loop_iteration"_s}, kSendToExternalProcess>(uint64_t{i});
        DevicePublish<TelemetryMetadata{"tt_bytes_processed_total"_s}, kSendToExternalProcess>(bytes);
        DevicePublish<TelemetryMetadata{"tt_progress"_s}, kSendToExternalProcess>(progress);
    }
}
