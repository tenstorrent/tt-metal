// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Positive: unpack-based untilize_init is deprecated in favour of pack_untilize.
#include "api/compute/common.h"
#include "api/compute/untilize.h"

void kernel_main() {
    untilize_init<false>(0);
    untilize_uninit(0);
}
