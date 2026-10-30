// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// loop with n iterations, 1 NOP
void l_n(unsigned n) { for (unsigned i = 0; i < n; i++) { TTI_NOP; } TTI_SETDMAREG(0, 9, 0, 0); }
// n+3 : start from 3
void l_from3(unsigned n) { for (unsigned i = 3; i < n; i++) { TTI_NOP; } }
// step 2
void l_step2(unsigned n) { for (unsigned i = 0; i < n; i += 2) { TTI_NOP; } }
// step 4 from 1
void l_step4(unsigned n) { for (unsigned i = 1; i < n; i += 4) { TTI_NOP; } }
// down count
void l_down(unsigned n) { for (unsigned i = n; i != 0; i--) { TTI_NOP; } }
// down count by 2 to 0 (n even)
void l_down2(unsigned n) { for (unsigned i = n; i > 0; i -= 2) { TTI_NOP; } }
// const 40
void l_c40() { for (unsigned i = 0; i < 40; i++) { TTI_NOP; } }
void l_c100() { for (unsigned i = 0; i < 100; i++) { TTI_NOP; } }
void l_c1000() { for (unsigned i = 0; i < 1000; i++) { TTI_NOP; } }
// two vars
void l_two(unsigned n, unsigned m) { for (unsigned i = m; i < n; i++) { TTI_NOP; } }
// nested
void l_nest(unsigned n, unsigned m) { for (unsigned i = 0; i < n; i++) { for (unsigned j = 0; j < m; j++) { TTI_NOP; } TTI_SETDMAREG(0, 5, 0, 0);} }
// loop with if inside
void l_if(unsigned n, unsigned a) { for (unsigned i = 0; i < n; i++) { TTI_NOP; if (a) { TTI_SETDMAREG(0, 5, 0, 0); } } }
// do-while
void l_dw(unsigned n) { unsigned i = 0; do { TTI_NOP; i++; } while (i < n); }
// while with complex cond
void l_cc(unsigned n, unsigned m) { unsigned i = 0; while (i < n && i < m) { TTI_NOP; i++; } }
