// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Publishes one Reading to each destination per iteration.
#include <cstdint>
#include "api/debug/device_publish.h"
#include "../external_types.hpp"

using namespace external_test;
using tt::tt_metal::experimental::streaming_profiler::kSendToExternalProcess;

void kernel_main() {
    for (uint32_t i = 0; i < N_ITERS; i++) {
        DevicePublish<Name{"metal"_s}>(Reading{0, i, reading_value(0, i)});
        DevicePublish<Name{"both"_s}, kSendToExternalProcess>(Reading{1, i, reading_value(1, i)});
    }
}
