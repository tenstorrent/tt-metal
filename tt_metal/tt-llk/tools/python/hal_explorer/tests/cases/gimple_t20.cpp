// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sfpi.h"
void sf1(unsigned n) {
    for (unsigned i = 0; i < n; i++) { sfpi::vFloat v = sfpi::dst_reg[0]; sfpi::dst_reg[0] = v + 1.0f; sfpi::dst_reg++; }
    TTI_NOP;
}
void sf2() {
    sfpi::dst_reg++; sfpi::dst_reg++;
    TTI_NOP;
}
