// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

inline void _sfpu_load_imm32_(const uint dest, const uint val)
{
    TT_SFPLOADI(dest, 10, (val & 0xFFFF));      // insmod == 10 will write the lower bits, and not affect the upper bits;
    TT_SFPLOADI(dest, 8, (val >> 16) & 0xFFFF); // insmod == 8 will write the upper bits, and not affect the lower bits;
}

inline void _sfpu_load_imm16_(const uint dest, const uint val)
{
    TT_SFPLOADI(dest, 2, val); // insmod == 2 will write imm16 value treated as unsigned integer, right justified and padded with zeroes on the MSBs
}

inline void _sfpu_load_config32_(const uint dest, const uint upper16, const uint lower16)
{
    // registers 11 through 14 are programmable "constants" which are shared across all 4 rows
    // They are updated only through the CONFIG path, which uses LREG[0] first and then copies it to the desired register location
    TTI_SFPLOADI(0, 10, lower16); // insmod == A will write the lower bits, and not affect the upper bits;
    TTI_SFPLOADI(0, 8, upper16);  // insmod == 8 will write the upper bits, and not affect the lower bits;
    TTI_SFPCONFIG(0, dest, 0);
}

inline void _init_sfpu_config_reg()
{
    _sfpu_load_config32_(0xF, 0x0, 0x0);
}

} // namespace sfpu
} // namespace ckernel
