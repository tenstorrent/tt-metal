// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_f() { for (std::uint32_t i = 0; i < 100; i++) { TT_SETC16(5, i); TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; } }
void f() { for (std::uint32_t i = 0; i < 100; i++) { TT_SETC16(5, 2 * i + 1); TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; } }
