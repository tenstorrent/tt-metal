// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common.h"

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

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
ALWI void srca_forced_exp_bfp2(uint32_t exp) {
#ifdef TRISC_UNPACK
    constexpr uint32_t bfp2_b = 15;  // DataFormat::Bfp2_b
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0x0f>(bfp2_b);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(1);
    cfg_reg_rmw_tensix<UNP0_FORCED_SHARED_EXP_shared_exp_RMW>(exp);
#endif
}

ALWI void srca_forced_exp_off() {
#ifdef TRISC_UNPACK
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(0);
#endif
}

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
