// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

inline void big_issue(const bool c) { if (!c) return; TTI_SETC16(0, 0);  TTI_SETC16(1, 1);  TTI_SETC16(2, 2);  TTI_SETC16(3, 3);  TTI_SETC16(4, 4);  TTI_SETC16(5, 5);  TTI_SETC16(6, 6);  TTI_SETC16(7, 7);  TTI_SETC16(8, 8);  TTI_SETC16(9, 9);  TTI_SETC16(10, 10);  TTI_SETC16(11, 11);  TTI_SETC16(12, 12);  TTI_SETC16(13, 13);  TTI_SETC16(14, 14);  TTI_SETC16(15, 15);  TTI_SETC16(16, 16);  TTI_SETC16(17, 17);  TTI_SETC16(18, 18);  TTI_SETC16(19, 19);  TTI_SETC16(20, 20);  TTI_SETC16(21, 21);  TTI_SETC16(22, 22);  TTI_SETC16(23, 23);  TTI_SETC16(24, 24);  TTI_SETC16(25, 25);  TTI_SETC16(26, 26);  TTI_SETC16(27, 27);  TTI_SETC16(28, 28);  TTI_SETC16(29, 29);  TTI_SETC16(30, 30);  TTI_SETC16(31, 31);  TTI_SETC16(0, 32);  TTI_SETC16(1, 33);  TTI_SETC16(2, 34);  TTI_SETC16(3, 35);  TTI_SETC16(4, 36);  TTI_SETC16(5, 37);  TTI_SETC16(6, 38);  TTI_SETC16(7, 39);  TTI_SETC16(8, 40);  TTI_SETC16(9, 41);  TTI_SETC16(10, 42);  TTI_SETC16(11, 43);  TTI_SETC16(12, 44);  TTI_SETC16(13, 45);  TTI_SETC16(14, 46);  TTI_SETC16(15, 47);  TTI_SETC16(16, 48);  TTI_SETC16(17, 49);  TTI_SETC16(18, 50);  TTI_SETC16(19, 51);  TTI_SETC16(20, 52);  TTI_SETC16(21, 53);  TTI_SETC16(22, 54);  TTI_SETC16(23, 55);  TTI_SETC16(24, 56);  TTI_SETC16(25, 57);  TTI_SETC16(26, 58);  TTI_SETC16(27, 59);  }
void caller_a(const bool c) { TTI_NOP; big_issue(c); }
void caller_b(const bool c) { big_issue(c); TTI_NOP; }
static void big_static(const std::uint32_t n) { for (std::uint32_t i = 0; i < n; ++i) { TT_SETC16(i, 1); TT_SETC16(i, 2); TT_SETC16(i, 3); TT_SETC16(i, 4); } }
void use_static_a(const std::uint32_t n) { big_static(n); big_static(n + 1); }
