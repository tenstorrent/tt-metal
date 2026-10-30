// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

extern volatile unsigned vflag;
// infinite loop with exit in the middle, code after loop
void l_break(unsigned a) {
    TTI_NOP;
    for (;;) { TTI_SETDMAREG(0, 1, 0, 0); if (vflag == a) break; TTI_SETDMAREG(0, 2, 0, 0); }
    TTI_SETDMAREG(0, 3, 0, 0);
}
// while(true) no exit
void l_inf() { while (true) { TTI_NOP; } }
// continue in middle
void l_cont(unsigned n, unsigned a) {
    for (unsigned i = 0; i < n; i++) {
        TTI_SETDMAREG(0, 1, 0, 0);
        if (vflag == a) continue;
        TTI_SETDMAREG(0, 2, 0, 0);
    }
    TTI_SETDMAREG(0, 3, 0, 0);
}
// break inside counted loop
void l_cbreak(unsigned n, unsigned a) {
    for (unsigned i = 0; i < n; i++) {
        TTI_SETDMAREG(0, 1, 0, 0);
        if (vflag == a) break;
        TTI_SETDMAREG(0, 2, 0, 0);
    }
    TTI_SETDMAREG(0, 3, 0, 0);
}
// poll on volatile
void l_poll() {
    while (vflag == 0) { TTI_NOP; }
    TTI_SETDMAREG(0, 3, 0, 0);
}
void l_poll2() {
    while (vflag == 0) { }
    TTI_SETDMAREG(0, 3, 0, 0);
}
// do while with continue
void l_dwc(unsigned n, unsigned a) {
    unsigned i = 0;
    do { i++; TTI_SETDMAREG(0, 1, 0, 0); if (a == i) continue; TTI_SETDMAREG(0, 2, 0, 0); } while (i < n);
    TTI_SETDMAREG(0, 3, 0, 0);
}
