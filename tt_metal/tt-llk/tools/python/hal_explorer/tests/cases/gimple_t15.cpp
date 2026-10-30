// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

extern volatile unsigned vflag;
// PHI of PHI + ternary merging words from several predecessors
void ph1(unsigned a, unsigned b, unsigned c) {
    unsigned w;
    if (a) w = 1; else if (b) w = 2; else if (c) w = 3; else w = 4;
    unsigned v = c ? w : w + 8;
    TT_SETDMAREG(0, v, 0, 0);
}
// merged via switch value
void ph2(unsigned a) {
    unsigned w;
    switch (a) { case 0: w = 5; break; case 1: w = 6; break; case 2: w = 9; break; default: w = a; }
    TT_SETDMAREG(0, w, 0, 0);
}
// word chosen by volatile read
void ph3() {
    unsigned w = vflag ? 7 : 8;
    TT_SETDMAREG(0, w, 0, 0);
}
// cond on a PHI value
void ph4(unsigned a, unsigned b) {
    unsigned w = a ? 1 : 0;
    if (b) w += 1;
    if (w == 2) { TTI_NOP; }
    TTI_SETDMAREG(0, 5, 0, 0);
}
