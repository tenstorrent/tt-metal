// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sfpi.h"
void sfpi_loop() { for (int d = 0; d < 8; ++d) { sfpi::vFloat x = sfpi::dst_reg[0]; sfpi::dst_reg[0] = x * x + 1.0f; sfpi::dst_reg++; } }
void sfpi_one() { sfpi::vFloat x = sfpi::dst_reg[0]; sfpi::dst_reg[1] = x * x; }
void sfpi_mixed() { TTI_NOP; sfpi::vFloat x = sfpi::dst_reg[0]; sfpi::dst_reg[0] = x * 2.0f; TTI_NOP; }
void tti_sfpu() { TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0); TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, 0); }
