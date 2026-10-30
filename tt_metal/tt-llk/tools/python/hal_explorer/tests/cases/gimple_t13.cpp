// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void f(unsigned a) { TTI_NOP; if (a) TTI_NOP; }
void f(unsigned a, unsigned b) { TTI_NOP; TTI_NOP; TTI_NOP; if (b) TTI_NOP; }
template <int N> void g() { for (int i = 0; i < N; i++) TTI_NOP; }
template void g<3>();
template void g<5>();
namespace ns { void h() { TTI_NOP; } }
struct S { static void m(unsigned a) { if (a) TTI_NOP; } };
void use(unsigned a) { S::m(a); }
template void g<7>();
void nl() {
    asm volatile("nop;\n\tnop\n" ::: "memory");
    TTI_NOP;
}
