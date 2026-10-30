// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#define N1 TTI_NOP
void sw8(unsigned a) {
    switch (a) {
    case 0: TTI_SETDMAREG(0, 10, 0, 0); break;
    case 1: TTI_SETDMAREG(0, 11, 0, 0); TTI_NOP; break;
    case 2: TTI_SETDMAREG(0, 12, 0, 0); TTI_NOP; TTI_NOP; break;
    case 3: TTI_SETDMAREG(0, 13, 0, 0); [[fallthrough]];
    case 4: TTI_SETDMAREG(0, 14, 0, 0); break;
    case 5: TTI_SETDMAREG(0, 15, 0, 0); break;
    case 6: TTI_SETDMAREG(0, 16, 0, 0); break;
    case 7: TTI_SETDMAREG(0, 17, 0, 0); break;
    default: break;
    }
    TTI_NOP;
}
