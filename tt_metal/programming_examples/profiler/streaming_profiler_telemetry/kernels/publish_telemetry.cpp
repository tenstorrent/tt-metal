// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Sends three telemetry metrics to the external process once per interval: tt_loop_iteration, the iteration it's on,
// tt_megabytes_processed_total, the megabytes (MiB) this core has handled so far, and tt_progress, the percent of its
// samples it has sent. The bytes are made up: each sample adds this core's runtime argument. A telemetry value is a
// uint8_t, uint16_t, uint32_t, uint64_t, float or double, so each call passes one of those, and each but the iteration
// count names its MetricUnit.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/telemetry_publish.h"

void kernel_main() {
    const uint32_t samples = get_arg_val<uint32_t>(0);
    const uint32_t interval_cycles = get_arg_val<uint32_t>(1);
    const uint32_t bytes_per_sample = get_arg_val<uint32_t>(2);
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < samples; i++) {
        riscv_wait(interval_cycles);
        bytes += bytes_per_sample;
        const double megabytes = static_cast<double>(bytes) / (1024.0 * 1024.0);
        const float progress = 100.0f * static_cast<float>(i + 1) / static_cast<float>(samples);
        TelemetryPublish("tt_loop_iteration", i);
        TelemetryPublish("tt_megabytes_processed_total", megabytes, MetricUnit::MEGABYTES);
        TelemetryPublish("tt_progress", progress, MetricUnit::PERCENT);
    }
}
