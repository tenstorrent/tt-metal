// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common.h"

#ifdef TRISC_MATH
#include "llk_math_transpose_dest_api.h"
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_A_api.h"
#endif

// =====================================================================================================
// Id-free (2.0) transpose_dest. Unlike the other 2.0 ops there is NO LLKOperand here: transpose_dest is a pure
// in-DST math op (it transposes a 32x32 tile already resident in the DST register), so it carries no L1 buffer,
// no data format, and no tile geometry -- it is ALREADY id-free. The only delta vs the legacy ckernel::
// transpose_dest is that the Blackhole path's `operand` argument (which is [[maybe_unused]] there) is dropped.
// The underlying LLK (llk_math_transpose_dest_api.h) is format-free and reused as-is. Blackhole only.
// =====================================================================================================

namespace ckernel {
namespace experimental {

#ifdef ARCH_BLACKHOLE

// clang-format off
/**
 * Paired init for transpose_dest. Reconfigures the math pipeline for the in-DST transpose op; call before
 * transpose_dest() (including when switching to it from another op). Set transpose_of_faces=false to run only
 * the inner face transpose used by 32-bit DST materialization paths.
 *
 * | Param Type | Name              | Description                                       | Type | Valid Range | Required |
 * |------------|-------------------|---------------------------------------------------|------|-------------|----------|
 * | Template   | is_32bit          | 32-bit DST materialization path                   | bool | N/A         | False    |
 * | Template   | transpose_of_faces| Also transpose the 2x2 face layout (full 32x32)   | bool | N/A         | False    |
 */
// clang-format on
template <bool is_32bit = false, bool transpose_of_faces = true>
ALWI void transpose_dest_init() {
    MATH((llk_math_transpose_dest_init<transpose_of_faces, is_32bit>()));
}

// clang-format off
/**
 * In-place 32x32 transpose *B[w,h] = A[h,w]* on the tile in DST[idst]. Set transpose_of_faces=false to run only
 * the inner face transpose used by 32-bit DST materialization paths. The DST register must be in the acquired
 * state. Blocking; compute-engine only. Pair with transpose_dest_init.
 *
 * | Param Type | Name              | Description                                     | Type     | Valid Range              | Required |
 * |------------|-------------------|-------------------------------------------------|----------|--------------------------|----------|
 * | Template   | is_32bit          | 32-bit DST materialization path                 | bool     | N/A                      | False    |
 * | Template   | transpose_of_faces| Also transpose the 2x2 face layout (full 32x32) | bool     | N/A                      | False    |
 * | Function   | idst              | Index of the tile in DST REG to transpose       | uint32_t | < acquired DST REG size  | True     |
 */
// clang-format on
template <bool is_32bit = false, bool transpose_of_faces = true>
ALWI void transpose_dest(std::uint32_t idst) {
    UNPACK((llk_unpack_set_srcb_dummy_valid()));
    MATH((llk_math_transpose_dest<transpose_of_faces, is_32bit>(idst)));
}

#endif  // ARCH_BLACKHOLE

}  // namespace experimental
}  // namespace ckernel
