// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// FLAGS: -fno-jump-tables
void sw8(const std::uint32_t m) {
    switch (m) {
        case 0: TTI_NOP; break;
        case 1: TTI_NOP; TTI_NOP; break;
        case 2: TTI_SETC16(1, 1); TTI_NOP; TTI_NOP; break;
        case 3: break;
        case 4: TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; break;
        case 5: TTI_SETC16(2, 2); break;
        case 6: TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; break;
        case 7: TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK); break;
        default: TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; break;
    }
    TTI_SETC16(3, 3);
}
void sw_sparse(const std::uint32_t m) {
    switch (m) {
        case 3: TTI_NOP; break;
        case 100: TTI_NOP; TTI_NOP; break;
        case 1000: TTI_SETC16(1, 1); TTI_NOP; TTI_NOP; break;
        case 5000: TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; break;
        case 9000: TTI_SETC16(2, 2); break;
    }
}
