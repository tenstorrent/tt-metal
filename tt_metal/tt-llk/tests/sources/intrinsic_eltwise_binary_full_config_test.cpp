
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* Full 3-thread compiler-config oracle: the same kernel as
 * intrinsic_eltwise_binary_test.cpp, but with all three TT_COMPILER_EMITS_*
 * defines set -- every thread's one-time setup is an author-written
 * config-declaration intrinsic:
 *
 *   UNPACK: __builtin_xttbh_unpack_hw_configure (config-declaration) + inline
 *           __builtin_xttbh_unpacr words (the per-tile data op, replacing the
 *           MOP)
 *   MATH:   __builtin_xttbh_math_hw_configure (config-declaration) +
 *           __builtin_xttbh_elwmul (the per-compute reconfig)
 *   PACK:   __builtin_xttbh_pack_hw_configure (config-declaration) + inline
 *           __builtin_xttbh_pacr words (the per-tile data op)
 *
 * The real-hardware golden: each thread's compiler-emitted config is
 * value-identical to the LLK's configure_* baseline (elfray-verified), and this
 * kernel exercises all three on silicon. */

#define TT_COMPILER_EMITS_UNPACK_CONFIG
#define TT_COMPILER_EMITS_MATH_CONFIG
#define TT_COMPILER_EMITS_PACK_CONFIG
#include "intrinsic_eltwise_binary_test.cpp"
