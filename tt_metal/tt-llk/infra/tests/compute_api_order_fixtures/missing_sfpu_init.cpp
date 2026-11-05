// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Positive: SFPU op with no preceding *_tile_init.
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/exp.h"

void kernel_main() {
    exp_tile<false>(0);
}
