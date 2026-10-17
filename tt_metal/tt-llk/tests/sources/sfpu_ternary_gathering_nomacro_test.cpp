// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// sfpu_ternary_test.cpp built the way tt-metal builds kernels under TT_METAL_ENABLE_GATHERING=1
// plus TT_METAL_DISABLE_SFPLOADMACRO=1: ENABLE_GATHERING is defined for the whole translation
// unit (driver, LLK headers and boot.h), so every replay record that goes through
// ckernel::load_replay_buf is bracketed with disable_gathering()/enable_gathering() and the boot
// code leaves gathering on; DISABLE_SFPLOADMACRO selects the SFPU kernels' plain-SFPLOAD paths.
//
// Build-only: test_gathering_config.py inspects the math ELF and never runs it.

#define ENABLE_GATHERING
#define DISABLE_SFPLOADMACRO

#include "sfpu_ternary_test.cpp"
