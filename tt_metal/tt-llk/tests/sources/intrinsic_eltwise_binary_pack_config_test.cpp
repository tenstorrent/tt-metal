
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* Pack-thread compiler-config oracle: the same kernel as
 * intrinsic_eltwise_binary_test.cpp, but with TT_COMPILER_EMITS_PACK_CONFIG
 * (and TT_COMPILER_EMITS_MATH_CONFIG, since the math thread's hw_configure is
 * now the author-written __builtin_xtt{wh,bh}_math_hw_configure declaration)
 * defined -- the pack thread's hardware configure switches from the LLK's
 * _llk_pack_hw_configure_wrapper_/_llk_pack_init_wrapper_ to the
 * compiler-managed __builtin_xttbh_pack_hw_configure config-declaration
 * intrinsic, and the per-tile data op becomes inline __builtin_xttbh_pacr
 * words (the MOP's PACR stream inlined).  The UNPACK thread is unaffected by
 * the defines. */

#define TT_COMPILER_EMITS_PACK_CONFIG
#define TT_COMPILER_EMITS_MATH_CONFIG
#include "intrinsic_eltwise_binary_test.cpp"
