// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Negative: SFPU op with its matching init.
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/exp.h"

void kernel_main() {
    exp_tile_init<false>();
    exp_tile<false>(0);
}
