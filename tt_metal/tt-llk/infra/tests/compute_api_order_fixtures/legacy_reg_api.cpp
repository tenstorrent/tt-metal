// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Positive: legacy DST acquire/release API is deprecated in favour of
// tile_regs_acquire()/tile_regs_release() (api/compute/reg_api.h).  These are
// not init functions, so they exercise the generic deprecated-api rule.
#include "api/compute/common.h"

void kernel_main() {
    acquire_dst();
    release_dst();
}
