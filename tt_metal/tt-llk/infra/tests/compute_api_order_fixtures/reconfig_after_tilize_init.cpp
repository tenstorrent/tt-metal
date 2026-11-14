// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Positive: issue #61146 - pack_reconfig_data_format after tilize_init disarms
// the tilize Z stride, corrupting the second half of every output tile.
#include "api/compute/common.h"
#include "api/compute/tilize.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    compute_kernel_hw_startup(0, 16);
    tilize_init<false>(0, 1, 16);
    pack_reconfig_data_format(16);
    tilize_block<false>(0, 1, 16);
    tilize_uninit(0, 16);
}
