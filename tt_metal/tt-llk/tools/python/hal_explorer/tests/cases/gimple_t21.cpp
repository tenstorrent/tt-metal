// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void nl() {
    asm volatile("nop;\n\tnop\n" ::: "memory");
    TTI_NOP;
    asm volatile("nop\n\tnop;\n\tnop" ::: "memory");
    TTI_NOP;
}
