
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* Unpack-thread compiler-config oracle: the same kernel as
 * intrinsic_eltwise_binary_test.cpp, but with TT_COMPILER_EMITS_UNPACK_CONFIG
 * (and TT_COMPILER_EMITS_MATH_CONFIG, since the math thread's hw_configure is
 * now the author-written __builtin_rvtt_{wh,bh}_math_hw_configure declaration)
 * defined, which switches the unpack thread's hardware configure from the LLK's
 * _llk_unpack_hw_configure_ to the compiler-managed
 * __builtin_rvtt_{wh,bh}_unpack_hw_configure config-declaration intrinsic.  The
 * functional oracle (torch golden vs L1 readback) then proves the compiler's
 * emitted unpack baseline is sufficient and correct -- the same kernel running
 * with the LLK configure is the control.  The PACK thread is unaffected by the
 * defines. */

#define TT_COMPILER_EMITS_UNPACK_CONFIG
#define TT_COMPILER_EMITS_MATH_CONFIG
#include "intrinsic_eltwise_binary_test.cpp"
