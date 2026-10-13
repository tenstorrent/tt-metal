// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// eltwise_unary_sfpu_test.cpp built the way tt-metal builds kernels under
// TT_METAL_ENABLE_GATHERING=1: ENABLE_GATHERING is defined for the whole translation unit
// (driver, LLK headers and boot.h), so load_replay_buf brackets every replay record with
// disable_gathering()/enable_gathering() and the boot code leaves gathering on.
//
// Build-only: test_gathering_config.py inspects the math ELF and never runs it.

#define ENABLE_GATHERING

#include "eltwise_unary_sfpu_test.cpp"
