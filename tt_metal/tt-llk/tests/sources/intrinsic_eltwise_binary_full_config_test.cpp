
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* Full 3-thread compiler-config oracle: the same kernel as
 * intrinsic_eltwise_binary_test.cpp, but with BOTH TT_COMPILER_EMITS_UNPACK_CONFIG
 * and TT_COMPILER_EMITS_PACK_CONFIG defined -- all three threads compute through
 * the compiler-managed Tensix intrinsics:
 *
 *   UNPACK: __builtin_xttbh_unpack_hw_configure (config-declaration) + inline
 *           __builtin_xttbh_unpacr words (the per-tile data op, replacing the
 *           MOP)
 *   MATH:   __builtin_xttbh_elwmul (ALU config from the compiler's baseline)
 *   PACK:   __builtin_xttbh_pack_hw_configure (config-declaration) + inline
 *           __builtin_xttbh_pacr words (the per-tile data op)
 *
 * The real-hardware golden: each thread's compiler-emitted config is
 * value-identical to the LLK's configure_* baseline (elfray-verified), and this
 * kernel exercises all three on silicon. */

#define TT_COMPILER_EMITS_UNPACK_CONFIG
#define TT_COMPILER_EMITS_PACK_CONFIG
#include "intrinsic_eltwise_binary_test.cpp"
