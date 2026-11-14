// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Positive: unpack init and execute use different broadcast configurations.
// This is the statically detectable form of the LLK_ASSERT unpack-config
// mismatches tracked by #39184 and #39945 (the hardware configuration and the
// init/execute arguments do not agree).
#include "api/compute/common.h"
#include "llk_unpack_AB_api.h"

void kernel_main() {
    llk_unpack_AB_init<BroadcastType::COL>(0, 1);
    llk_unpack_AB<BroadcastType::ROW>(0, 1, 0, 0);
}
