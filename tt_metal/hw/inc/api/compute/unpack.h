// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common_globals.h"
#include "sentinel/compute_kernel_sentinel.h"
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Short init for the binary (AB) unpacker. Configures the unpacker to read a pair
 * of operands from the specified input circular buffers with the given broadcast
 * and transpose modes.
 *
 * | Argument  | Description                                                  | Type      | Valid Range            | Required |
 * |-----------|--------------------------------------------------------------|-----------|------------------------|----------|
 * | icb0      | The identifier of the circular buffer (CB) containing A      | uint32_t  | 0 to 31                | True     |
 * | icb1      | The identifier of the circular buffer (CB) containing B      | uint32_t  | 0 to 31                | True     |
 * | transpose | Transpose mode to apply during unpack                        | Transpose | Any Transpose value    | False    |
 */
// clang-format on
template <BroadcastType bcast_type = BroadcastType::NONE>
ALWI void unpack_AB_init_short(uint32_t icb0, uint32_t icb1, Transpose transpose = Transpose::None) {
    UNPACK((llk_unpack_AB_init<bcast_type>(icb0, icb1, transpose)));
}

}  // namespace ckernel
