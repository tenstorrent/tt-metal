// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void r1(unsigned n) { lltt::replay(3, n); }
void r2(unsigned n, bool a) { if (a) lltt::replay(3, n); else lltt::replay(4, 7); TTI_NOP; }
void r3(unsigned n) { lltt::record<lltt::Exec>(2, n); TTI_NOP; TTI_NOP; }
void r4() { lltt::replay(0, 5); lltt::replay(31, 32); lltt::record(1, 8); }
void r5(unsigned n) { lltt::replay(1, n & 7); lltt::replay(1, n + 1); }
