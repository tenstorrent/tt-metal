// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

namespace ns {
void reference_x() { TTI_NOP; }
void x() { TTI_STALLWAIT(1,2); }
}
// static / inline rewrite is not emitted
void reference_s() { TTI_NOP; }
static void s() { TTI_STALLWAIT(1,2); }
void reference_i() { TTI_NOP; }
inline void i() { TTI_STALLWAIT(1,2); }
// typo: no counterpart
void reference_typo() { TTI_NOP; }
void typo2() { TTI_STALLWAIT(1,2); }
// overload
void reference_o(int a) { TTI_NOP; }
void o(int a) { TTI_NOP; }
void reference_o(float a) { TTI_NOP; TTI_NOP; }
void o(float a) { TTI_NOP; }
// template
template <int N> void reference_t() { for (int k = 0; k < N; ++k) TTI_NOP; }
template <int N> void t() { TTI_NOP; }
template void reference_t<3>();
template void t<3>();
