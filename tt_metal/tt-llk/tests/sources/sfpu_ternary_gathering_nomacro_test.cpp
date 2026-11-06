// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// sfpu_ternary_test.cpp with ENABLE_GATHERING and DISABLE_SFPLOADMACRO defined for the whole
// translation unit, as tt-metal builds kernels under TT_METAL_ENABLE_GATHERING=1 plus
// TT_METAL_DISABLE_SFPLOADMACRO=1. Build-only: test_gathering_config.py inspects the math ELF.

#define ENABLE_GATHERING
#ifndef DISABLE_SFPLOADMACRO // the harness defines it as 1 under --disable-sfploadmacro
#define DISABLE_SFPLOADMACRO
#endif

#include "sfpu_ternary_test.cpp"
