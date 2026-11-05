// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Warning: the init does not depend on the loop variable and is re-run every
// iteration.
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    for (uint32_t i = 0; i < 4; ++i) {
        add_init(0, 1);
        add_tiles(0, 1, i, i, 0);
    }
}
