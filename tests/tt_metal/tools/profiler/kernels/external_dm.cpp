// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Publishes one Reading to each destination per iteration, or with PROBE one "probe" reading for the external process.
#include <cstdint>
#include "api/debug/device_publish.h"
#include "../records_types.hpp"

using namespace external_test;
using tt::tt_metal::experimental::device_observability::kSendToExternalProcess;
using tt::tt_metal::experimental::device_observability::Timestamp;

void kernel_main() {
#ifdef PROBE
    DevicePublish<Name{"probe"_s}, kSendToExternalProcess>(Reading{Timestamp::now(), 1, 0, 0});
#else
    for (uint32_t i = 0; i < N_ITERS; i++) {
        DevicePublish<Name{"metal"_s}>(Reading{Timestamp::now(), 0, i, reading_value(0, i)});
        DevicePublish<Name{"both"_s}, kSendToExternalProcess>(Reading{Timestamp::now(), 1, i, reading_value(1, i)});
    }
#endif
}
