// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// condition on loop-varying value; expected: only iteration 2 issues S2
void l_ivcond(unsigned n) {
    for (unsigned i = 0; i < n; i++) {
        TTI_SETDMAREG(0, 1, 0, 0);
        if (i == 2) { TTI_SETDMAREG(0, 2, 0, 0); }
    }
}
// constant 8 iterations with iv condition (GCC may unroll fully)
void l_ivcond8() {
    for (unsigned i = 0; i < 8; i++) {
        TTI_SETDMAREG(0, 1, 0, 0);
        if (i == 2) { TTI_SETDMAREG(0, 2, 0, 0); }
    }
}
// runtime word depending on iv  (instruction word = base + i)
void l_ivword(unsigned n) {
    for (unsigned i = 0; i < n; i++) {
        TT_SETDMAREG(0, i, 0, 0);
    }
}
void l_ivword40() {
    for (unsigned i = 0; i < 40; i++) {
        TT_SETDMAREG(0, i, 0, 0);
    }
}
