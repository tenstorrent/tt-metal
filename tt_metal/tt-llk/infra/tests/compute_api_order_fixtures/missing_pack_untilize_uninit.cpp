// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Positive: pack_untilize_init arms packer state; pack_untilize_uninit is required.
#include "api/compute/common.h"
#include "api/compute/pack_untilize.h"

void kernel_main() {
    pack_untilize_init<8, 8>(0, 16);
    pack_untilize_block(0, 1, 16);
}
