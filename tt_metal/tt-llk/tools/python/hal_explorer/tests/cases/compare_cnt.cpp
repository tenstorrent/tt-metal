// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void nest(const std::uint32_t n, const std::uint32_t m) { TTI_NOP; for (std::uint32_t i = 0; i < n; i++) { TTI_NOP; TTI_NOP; for (std::uint32_t j = 0; j < m; j++) { TTI_STALLWAIT(1,2); } } TTI_NOP; }
void step(const std::uint32_t n) { for (std::uint32_t i = 0; i < n; i += 3) { TTI_NOP; } }
void down(const std::uint32_t n) { for (std::uint32_t i = n; i != 0; i -= 2) { TTI_NOP; TTI_NOP; } }
void sub(const std::uint32_t n) { for (std::uint32_t i = 2; i < n; i++) { TTI_NOP; } }
void unrolled() { for (std::uint32_t i = 0; i < 6; i++) { TTI_NOP; } }
void big() { for (std::uint32_t i = 0; i < 100000; i++) { TTI_NOP; TTI_STALLWAIT(1,2); } }
