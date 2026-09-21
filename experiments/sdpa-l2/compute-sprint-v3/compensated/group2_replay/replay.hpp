// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#if defined(TRISC_PACK) || defined(TRISC_MATH)
namespace ckernel::sfpu {
inline void init_group2_identity_replay() {
    // Same root-add and BF16 round/store templates as the frozen compensation.
    init_sprint_compensated_block_macros();
    // DST: hi0,hi1,lo0,lo1,local0,local1,chunk0,chunk1.
    // Macro outputs root0/root1=L1/L2; full totals=L0/L3.
    TTI_REPLAY(0, 18, 0, 1);
    TTI_SFPLOAD(0, 0, ADDR_MOD_6, 0);
    TTI_SFPLOAD(3, 0, ADDR_MOD_6, 64);
    TTI_SFPLOADMACRO(9, 0, ADDR_MOD_6, 128);
    TTI_SFPLOADMACRO(14, 0, ADDR_MOD_6, 192);
    TTI_SFPLOAD(4, 0, ADDR_MOD_6, 256);
    TTI_SFPLOAD(5, 0, ADDR_MOD_6, 320);
    TTI_SFPLOAD(6, 0, ADDR_MOD_6, 384);
    TTI_SFPLOAD(7, 0, ADDR_MOD_6, 448);
    TTI_SFPADD(10, 4, 6, 4, 0);
    TTI_SFPADD(10, 5, 7, 5, 0);
    TTI_SFPADD(10, 1, 4, 0, 0);
    TTI_SFPADD(10, 2, 5, 3, 0);
    TTI_SFPLOADMACRO(1, 0, ADDR_MOD_6, 0);
    TTI_SFPLOADMACRO(6, 0, ADDR_MOD_6, 64);
    TTI_SFPADD(10, 0, 1, 0, 2);
    TTI_SFPADD(10, 3, 2, 3, 2);
    TTI_SFPLOADMACRO(1, 0, ADDR_MOD_6, 128);
    TTI_SFPLOADMACRO(6, 0, ADDR_MOD_7, 192);
}
inline void calculate_group2_identity_replay() {
#pragma GCC unroll 8
    for (int i = 0; i < 32; ++i) {
        TTI_REPLAY(0, 18, 0, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}
} // namespace ckernel::sfpu
#endif
