// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// eltwise_unary_sfpu_test.cpp built the way tt-metal builds Blackhole kernels by default (gathering
// on): ENABLE_GATHERING is defined for the whole translation unit
// (driver, LLK headers and boot.h), so load_replay_buf brackets every replay record with
// disable_gathering()/enable_gathering() and the boot code leaves gathering on.
//
// Build-only: test_gathering_config.py inspects the math ELF and never runs it.

#ifndef ENABLE_GATHERING
#define ENABLE_GATHERING
#endif

#include "eltwise_unary_sfpu_test.cpp"
