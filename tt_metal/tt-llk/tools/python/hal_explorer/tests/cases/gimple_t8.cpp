// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

extern volatile unsigned vflag;
void m1(unsigned n) {
    for (unsigned i = 0; i < n; i++) {
        TTI_SETDMAREG(0, 1, 0, 0);
        if (vflag == i) break;
        TTI_SETDMAREG(0, 2, 0, 0);
        if (vflag == 7) break;
        TTI_SETDMAREG(0, 4, 0, 0);
    }
    TTI_SETDMAREG(0, 3, 0, 0);
}
void m2(unsigned a) {
    // exit at top, unconditional back edge
    TTI_NOP;
    while (true) {
        if (vflag == a) goto out;
        TTI_SETDMAREG(0, 1, 0, 0);
        if (vflag == 3) goto out;
        TTI_SETDMAREG(0, 2, 0, 0);
    }
out:
    TTI_SETDMAREG(0, 3, 0, 0);
}
void m3(unsigned a) {
    unsigned i = 0;
again:
    TTI_SETDMAREG(0, 1, 0, 0);
    if (vflag == a) { TTI_SETDMAREG(0, 2, 0, 0); goto again; }
    if (vflag == 9) { TTI_SETDMAREG(0, 5, 0, 0); goto again; }
    TTI_SETDMAREG(0, 3, 0, 0);
}
