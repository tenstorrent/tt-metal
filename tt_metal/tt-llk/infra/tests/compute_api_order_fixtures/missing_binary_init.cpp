// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Positive: eltwise binary op with no preceding init.
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    compute_kernel_hw_startup(0, 1, 16);
    add_tiles(0, 1, 0, 0, 0);
}
