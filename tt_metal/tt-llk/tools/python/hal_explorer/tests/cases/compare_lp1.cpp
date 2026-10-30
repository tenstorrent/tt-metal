// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_f(const std::uint32_t n, const std::uint32_t flag) { for (std::uint32_t i = 0; i < n; i++) { if (flag) continue; TTI_NOP; } TTI_STALLWAIT(1,2); }
void f(const std::uint32_t n, const std::uint32_t flag) { for (std::uint32_t i = 0; i < n; i++) { if (flag) continue; TTI_NOP; } TTI_STALLWAIT(1,3); }
void reference_g(const std::uint32_t n, const std::uint32_t m) { for (std::uint32_t i = 0; i < n; i++) { for (std::uint32_t j = 0; j < m; j++) { TTI_NOP; } } TTI_STALLWAIT(1,2); }
void g(const std::uint32_t n, const std::uint32_t m) { for (std::uint32_t i = 0; i < n; i++) { for (std::uint32_t j = 0; j < m; j++) { TTI_NOP; } } TTI_STALLWAIT(1,3); }
void reference_h(const std::uint32_t n) { while (true) { TTI_NOP; if (n) break; } TTI_STALLWAIT(1,2); }
void h(const std::uint32_t n) { while (true) { TTI_NOP; if (n) break; } TTI_STALLWAIT(1,3); }
