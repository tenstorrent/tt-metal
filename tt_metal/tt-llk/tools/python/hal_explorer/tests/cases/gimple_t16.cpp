// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

[[noreturn]] void die();
void tri(unsigned n) {
    for (unsigned i = 0; i < n; i++)
        for (unsigned j = 0; j < i; j++)
            TTI_NOP;
}
void trap1(unsigned a) {
    TTI_SETDMAREG(0, 1, 0, 0);
    if (a) __builtin_trap();
    TTI_SETDMAREG(0, 2, 0, 0);
}
void trap2(unsigned a) {
    TTI_SETDMAREG(0, 1, 0, 0);
    if (a > 4) die();
    TTI_SETDMAREG(0, 2, 0, 0);
}
void ext(unsigned);
void call1(unsigned a) {
    TTI_SETDMAREG(0, 1, 0, 0);
    ext(a);
    TTI_SETDMAREG(0, 2, 0, 0);
}
