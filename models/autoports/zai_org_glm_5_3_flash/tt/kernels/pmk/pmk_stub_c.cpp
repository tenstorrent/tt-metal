// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK stub phase body (TRISC compute): see pmk_stub_dm.cpp. Compute kernels of this tree are plain `void kernel_main()`
// (no NAMESPACE / MAIN), as kernels/mk/owner_c.cpp composes them.
#include <cstdint>

#include "api/compute/common.h"

void kernel_main() {
    (void)PH_CT0;
    (void)ph_rt0;
}
