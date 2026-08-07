// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

//
// DEPRECATED HEADER.
//
// The in-DST transpose compute API has been renamed to drop the `_wh` suffix. The canonical API now
// lives in "api/compute/transpose_dest.h" (transpose_dest_init / transpose_dest).
//
// Everything below is a thin [[deprecated]] compatibility shim that forwards to the new API. This
// whole file is scheduled for removal (see .github/deprecations.json); migrate to transpose_dest.h:
//   transpose_wh_dest_init_short<is_32bit, transpose_of_faces>(operand) ->  transpose_dest_init<is_32bit,
//   transpose_of_faces>(operand); transpose_wh_dest<is_32bit, transpose_of_faces>(idst)        ->
//   transpose_dest<is_32bit, transpose_of_faces>(idst);
//

#include "api/compute/transpose_dest.h"

namespace ckernel {

/**
 * @deprecated Use transpose_dest_init<is_32bit, transpose_of_faces>(). See "api/compute/transpose_dest.h".
 */
template <bool is_32bit = false, bool transpose_of_faces = true>
[[deprecated("Use transpose_dest_init<is_32bit, transpose_of_faces>(). See api/compute/transpose_dest.h.")]] ALWI void
transpose_wh_dest_init_short([[maybe_unused]] uint32_t operand = 0) {
    transpose_dest_init<is_32bit, transpose_of_faces>(operand);
}

/**
 * @deprecated Use transpose_dest<is_32bit, transpose_of_faces>(idst). See "api/compute/transpose_dest.h".
 */
template <bool is_32bit = false, bool transpose_of_faces = true>
[[deprecated("Use transpose_dest<is_32bit, transpose_of_faces>(idst). See api/compute/transpose_dest.h.")]] ALWI void
transpose_wh_dest(uint32_t idst) {
    transpose_dest<is_32bit, transpose_of_faces>(idst);
}

}  // namespace ckernel
