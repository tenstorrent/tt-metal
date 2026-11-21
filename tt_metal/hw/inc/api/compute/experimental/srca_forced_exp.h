// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common.h"
// Blackhole-only: other archs' llk_api entry points reject these calls (LLK_ASSERT).
#ifdef TRISC_UNPACK
#include "experimental/llk_unpack_srca_forced_exp_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Makes the SrcA unpacker (in1 of a matmul) read BFP2_b tiles that hold only mantissas: the input format becomes
 * BFP2_b (one sign bit and one magnitude bit per datum) and every datum takes the shared exponent exp in place of the
 * tile's exponent section, so a tile is 256 bytes. With exp = 127 a datum is -1, 0 or +1 (sign 1 with magnitude 0
 * reads as -inf). The products with bf16 activations are exact at HiFi3 and HiFi4.
 *
 * Call after the init that configures SrcA's format (matmul_init, reconfig_data_format_srca), which this overrides;
 * srca_forced_exp_off clears the forced exponent before SrcA reads a tile of another format.
 *
 * | Argument | Description                               | Type     | Valid Range | Required |
 * |----------|-------------------------------------------|----------|-------------|----------|
 * | exp      | The exponent every datum takes (biased)   | uint32_t | 0 to 255    | True     |
 */
// clang-format on
ALWI void srca_forced_exp_bfp2(uint32_t exp) { UNPACK((llk_unpack_srca_forced_exp_bfp2(exp))); }

ALWI void srca_forced_exp_off() { UNPACK((llk_unpack_srca_forced_exp_off())); }

}  // namespace ckernel
