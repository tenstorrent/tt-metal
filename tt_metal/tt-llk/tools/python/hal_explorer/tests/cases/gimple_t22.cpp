// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void q1(unsigned char a, signed char s, bool b, short h) {
    if (a == 3) TTI_NOP;
    if (s < -2) { TTI_NOP; TTI_NOP; }
    if (b && h > 100) { TTI_SETDMAREG(0,1,0,0); }
    if ((a & 4) && (h & 1)) { TTI_SETDMAREG(0,2,0,0); }
    if (a > 7 || h < 0) { TTI_SETDMAREG(0,3,0,0); }
}
void q2(unsigned a) {
    // 16-step chain to exceed text depth
    unsigned w = a;
    w = (w ^ 1) * 3 + 7; w = (w ^ 2) * 3 + 7; w = (w ^ 3) * 3 + 7; w = (w ^ 4) * 3 + 7;
    w = (w ^ 5) * 3 + 7; w = (w ^ 6) * 3 + 7; w = (w ^ 7) * 3 + 7; w = (w ^ 8) * 3 + 7;
    TT_SETDMAREG(0, w & 0x3fff, 0, 0);
}
