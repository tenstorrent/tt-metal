// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

extern volatile unsigned vflag;
void b1(unsigned n) { unsigned m = n * 3; for (unsigned i = 0; i < m; i++) TTI_NOP; }
void b2(unsigned n) { unsigned m = n * 3; for (unsigned i = 0; i < m; i += 2) TTI_NOP; }
void b3(unsigned n) { unsigned m = n + vflag; for (unsigned i = 0; i < m; i += 2) TTI_NOP; }
void b4(unsigned n) { unsigned m = n + vflag; for (unsigned i = 3; i < m; i += 1) TTI_NOP; }
void b5(unsigned n) { unsigned m = n + vflag; unsigned i = 0; do { TTI_NOP; i += 4; } while (i < m); }
void b6(unsigned n) { unsigned m = n + vflag; for (unsigned i = m; i > 2; i -= 2) TTI_NOP; }
void b7(int n) { for (int i = 0; i < n; i += 3) TTI_NOP; }
void b8(unsigned n) { unsigned m = n + vflag; for (unsigned i = 0; i != m; i += 2) TTI_NOP; }
