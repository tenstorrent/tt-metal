// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Negative: broadcast with the matching uninit.
#include "api/compute/common.h"
#include "api/compute/bcast.h"

void kernel_main() {
    unary_bcast_init<BroadcastType::COL>(0);
    unary_bcast<BroadcastType::COL>(0, 0, 0);
    unary_bcast_uninit<BroadcastType::COL>(0);
}
