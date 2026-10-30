// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void f1(unsigned a, unsigned b) {
    TTI_NOP;
    if (a) { TTI_SETDMAREG(0, 1, 0, 0); } else { TTI_SETDMAREG(0, 2, 0, 0); TTI_NOP; }
    if (b) { TTI_SETDMAREG(0, 3, 0, 0); }
    TTI_NOP;
}
