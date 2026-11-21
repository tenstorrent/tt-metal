// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Positive: broadcast init arms persistent unpack/math state; unary_bcast_uninit
// is required before the kernel returns.
#include "api/compute/common.h"
#include "api/compute/bcast.h"

void kernel_main() {
    unary_bcast_init<BroadcastType::COL>(0);
    unary_bcast<BroadcastType::COL>(0, 0, 0);
}
