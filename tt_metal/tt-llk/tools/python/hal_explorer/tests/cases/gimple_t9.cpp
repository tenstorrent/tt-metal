// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// each expected count in comment (iterations of 1 NOP)
void c1() { for (unsigned i = 0; i < 100; i += 3) TTI_NOP; }       // 34
void c2() { for (unsigned i = 1; i < 100; i += 4) TTI_NOP; }       // 25
void c3() { for (unsigned i = 100; i > 0; i -= 7) TTI_NOP; }       // 15
void c4() { for (unsigned i = 0; i <= 99; i++) TTI_NOP; }          // 100
void c5() { for (int i = -50; i < 50; i++) TTI_NOP; }              // 100
void c6() { for (int i = 99; i >= 0; i--) TTI_NOP; }               // 100
void c7() { for (unsigned i = 0; i < 100; i += 2) TTI_NOP; }       // 50
void c8() { for (unsigned i = 5; i < 105; i++) TTI_NOP; }          // 100
void c9() { for (unsigned i = 0; i < 100; i++) { TTI_NOP; TTI_NOP; } } // 200
void c10() { for (unsigned i = 0; i < 100; i++) { for (unsigned j = 0; j < 100; j++) TTI_NOP; } } // 10000
void c11() { for (unsigned i = 0; i < 50; i++) { for (unsigned j = 0; j < 3; j++) TTI_NOP; TTI_SETDMAREG(0,1,0,0);} } // 200
void c12() { for (unsigned char i = 0; i < 200; i++) TTI_NOP; }    // 200
void c13() { for (unsigned i = 0; i != 100; i+=5) TTI_NOP; }       // 20
void c14() { unsigned i = 0; do { TTI_NOP; } while (++i < 77); }   // 77
void c15() { for (unsigned i = 0; i < 1000000; i++) TTI_NOP; }   // 1000000
