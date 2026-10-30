// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void p1(bool a) { TT_SETDMAREG(0, a ? 5 : 7, 0, 0); }
void p2(unsigned a, unsigned b) { TT_SETDMAREG(0, (a & 0xf) | (b << 4), 0, 0); }
void p3(unsigned a) { TT_SETDMAREG(0, a, 0, 0); }
void p4(unsigned a) { TT_SETDMAREG(1, a ^ 3, 0, a & 1); }
void p5(unsigned char a) { TT_SETDMAREG(0, a, 1, 3); }
void p6(unsigned a) { TT_SETDMAREG(0, 5, 0, a + 2); }
void p7(unsigned a, unsigned b) { unsigned w = a ? 1 : (b ? 2 : 3); TT_SETDMAREG(0, w, 0, 0); }
void p8(unsigned a, unsigned b) { unsigned w = a; if (b) w = w + 4; TT_SETDMAREG(0, w, 0, 0); }
void p9(unsigned a) { TT_SETDMAREG(0, a >> 3, 0, 0); }
void p10(unsigned a) { TT_SETDMAREG(0, (a & 0x3) << 2, 0, 0); }
void p11(unsigned a) { TT_SETDMAREG(0, 5, 0, (a & 0x1) ? 4 : 9); }
void p12(unsigned a) { TT_SETDMAREG(0, 5, 0, -a); }
void p13(unsigned a) { TT_SETDMAREG(0, 5, 0, ~a & 3); }
void p14(unsigned a) { TT_SETDMAREG(0, 5, 0, a * 4); }
void p15(unsigned a, unsigned b) { TT_SETDMAREG(0, 5, 0, a - b); }
