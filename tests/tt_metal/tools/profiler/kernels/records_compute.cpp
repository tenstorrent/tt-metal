// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Each iteration on the compute RISCs publishes two records the data-movement RISCs also publish, so records from all
// five RISCs land in the same callback spans.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/debug/device_publish.h"
#include "../records_types.hpp"

using namespace records_test;

void kernel_main() {
    for (uint32_t i = 0; i < N_ITERS; i++) {
        DevicePublish<Name{"tiles"_s}>(i, float(i) * 0.5f);
        DevicePublish<Name{"start"_s}>();
    }
}
