// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Minimal repro for an SFPI compiler ICE: SFPU code in a TU compiled with the shared PCH and
// without LTO. ComputeConfig::enable_trisc2_rvv is just the vehicle that gets TRISC2 (pack)
// compiled with -fno-lto; the vector extension itself plays no part (no RVV code here).
//
// With the knob off TRISC2 is compiled with -flto and the same source builds fine.
//
// Compile-only test collateral (tests/tt_metal/tt_metal/jit_build/test_trisc2_rvv.cpp);
// never dispatched to a device.

#include <cstdint>

#include "api/compute/common.h"

#ifdef TRISC_PACK
#include "sfpi.h"
#endif

void kernel_main() {
#ifdef TRISC_PACK
    // Materializing an SFPU immediate is enough: rvtt_immvar_expand turns it into an SFPLOADI.
    sfpi::vFloat v = 1.5f;
    sfpi::dst_reg[0] = v;
#endif
}
