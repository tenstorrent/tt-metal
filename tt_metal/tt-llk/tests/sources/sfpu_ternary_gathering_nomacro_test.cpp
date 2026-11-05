// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// sfpu_ternary_test.cpp with ENABLE_GATHERING and DISABLE_SFPLOADMACRO defined for the whole
// translation unit, as tt-metal builds Blackhole kernels by default plus
// TT_METAL_DISABLE_SFPLOADMACRO=1. Build-only: test_gathering_config.py inspects the math ELF.

#ifndef ENABLE_GATHERING // the harness defines it on Blackhole unless TT_METAL_DISABLE_GATHERING=1
#define ENABLE_GATHERING
#endif
#ifndef DISABLE_SFPLOADMACRO // the harness defines it as 1 under --disable-sfploadmacro
#define DISABLE_SFPLOADMACRO
#endif

#include "sfpu_ternary_test.cpp"
