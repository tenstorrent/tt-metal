// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Negative: tilize_init / tilize_block / tilize_uninit.
#include "api/compute/common.h"
#include "api/compute/tilize.h"

void kernel_main() {
    compute_kernel_hw_startup(0, 16);
    tilize_init<false>(0, 1, 16);
    tilize_block<false>(0, 1, 16);
    tilize_uninit(0, 16);
}
