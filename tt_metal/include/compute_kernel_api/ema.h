// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_ema_sfpu_entry.h"
#endif

namespace ckernel {
/**
 * @brief Performs a ema's online algorithm update for mean and m2 on a tile in the DST register.
 *
 * This operation computes the running mean and m2 for a stream of data, enabling numerically stable
 * calculation of statistics in a single pass. The DST register buffer must be in acquired state via @ref
 * tile_regs_acquire call. This call is blocking and is only available on the compute engine.
 *
 * @tparam input_dst_index   The index of the tile in DST register buffer containing the new input.
 *                           Must be less than the size of the DST register.
 * @param first_sample       Whether this is the first sample in the stream.
 * @note All TILE_WIDTH (32) columns of the input tile are processed by this function.
 *
 * @return None. Mean and m2 tiles are updated in place.
 */

template <uint32_t input_dst_index>
ALWI void ema_tile(bool first_sample) {
    MATH((llk_math_ema_sfpu<input_dst_index>(first_sample)));
}

/**
 * Uses a copy of the ternery_sfpu_init
 * Programs the replay for fast LLK. Needed otherwise LLK wont work
 */
ALWI void ema_init() { MATH((llk_math_ema_sfpu_init())); }

}  // namespace ckernel
