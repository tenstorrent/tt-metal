// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Negative: eltwise binary init followed by its operation.
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    compute_kernel_hw_startup(0, 1, 16);
    add_init(0, 1);
    add_tiles(0, 1, 0, 0, 0);
}
