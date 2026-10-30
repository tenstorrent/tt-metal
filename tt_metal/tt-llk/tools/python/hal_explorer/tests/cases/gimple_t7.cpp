// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

extern volatile unsigned vflag;
// word after loop depends on loop-exit value of i; expected: SETDMAREG with sigsel = (trip count) -> rt
void after_loop() {
    unsigned i = 0;
    while (vflag != 5) { TTI_NOP; i++; }
    TT_SETDMAREG(0, i, 0, 0);
}
void after_loop2(unsigned n) {
    unsigned w = 3;
    for (unsigned i = 0; i < n; i++) { TTI_NOP; w = w * 2 + vflag; }
    TT_SETDMAREG(0, w, 0, 0);
}
// reduction accumulate and use after loop with fixed trip
void after_loop3() {
    unsigned w = 3;
    for (unsigned i = 0; i < 100; i++) { TTI_NOP; w = w * 2 + vflag; }
    TT_SETDMAREG(0, w, 0, 0);
}
