// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

extern volatile unsigned vflag;
void u64c() { for (unsigned long long i = 0; i < 300; i++) TTI_NOP; }
void u16c() { for (unsigned short i = 0; i < 300; i++) TTI_NOP; }
void u8c() { for (unsigned char i = 0; i < 250; i++) TTI_NOP; }
void u8c2() { for (unsigned char i = 10; i < 250; i += 3) TTI_NOP; }
void i64n(unsigned long long n) { for (unsigned long long i = 0; i < n; i++) TTI_NOP; }
void dwn(int n) { for (int i = n; i > 0; i--) TTI_NOP; }
void s8c() { for (signed char i = -100; i < 100; i++) TTI_NOP; }
