// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Sends two metrics to the external process once per interval: loop_iteration, the iteration it's on, and
// bytes_processed_total, the bytes this core has handled so far. The bytes are made up: each sample adds this core's
// runtime argument.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/device_publish.h"
#include "hostdevcommon/sample.hpp"

void kernel_main() {
    using namespace tt::tt_metal::experimental::streaming_profiler;
    const uint32_t samples = get_arg_val<uint32_t>(0);
    const uint32_t interval_cycles = get_arg_val<uint32_t>(1);
    const uint32_t bytes_per_sample = get_arg_val<uint32_t>(2);
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < samples; i++) {
        riscv_wait(interval_cycles);
        bytes += bytes_per_sample;
        DevicePublish<SampleMetadata{"loop_iteration"_s}, kSendToExternalProcess>(i);
        DevicePublish<SampleMetadata{"bytes_processed_total"_s}, kSendToExternalProcess>(bytes);
    }
}
