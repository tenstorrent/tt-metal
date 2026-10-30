// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

extern volatile unsigned vflag;
void w1() { while (vflag != 1) { while (vflag != 2) { TTI_NOP; } } TTI_SETDMAREG(0,1,0,0); }
void w2() { while (vflag != 1) { TTI_SETDMAREG(0,2,0,0); while (vflag != 2) { TTI_NOP; } } TTI_SETDMAREG(0,1,0,0); }
void w3(unsigned n) { for (unsigned i = 0; i < n; i++) { while (vflag != i) { TTI_NOP; } } }
void w4(unsigned n) { for (unsigned i = 0; i < n; i++) { for (unsigned j = 0; j < i; j++) TTI_NOP; TTI_SETDMAREG(0,2,0,0);} }
void w5(unsigned n) { for (unsigned i = 1; i <= n; i++) { for (unsigned j = i; j < n; j++) TTI_NOP; } }
