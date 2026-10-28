// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Publishes N_RECORDS records counting up from 0, then N_BIG of the largest records, enough to fill the host's FIFO
// many times over while a callback isn't reading it.
#include <cstdint>
#include "api/debug/device_publish.h"
#include "../records_types.hpp"

using namespace records_test;

// Too large for the stack.
Big big;

void kernel_main() {
    for (uint32_t i = 0; i < N_RECORDS; i++) {
        DevicePublish<Name{"count"_s}>(i);
    }
    for (uint32_t i = 0; i < N_BIG; i++) {
        big.words[0] = i;
        DevicePublish<Name{"big"_s}>(big);
    }
}
