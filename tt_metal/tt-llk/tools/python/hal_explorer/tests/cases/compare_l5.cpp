// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// equivalent trip counts, different loop shapes
void reference_f(const std::uint32_t n) { for (std::uint32_t i = 0; i < n; i++) { TTI_NOP; } }
void f(const std::uint32_t n)           { for (std::uint32_t i = n; i > 0; i--) { TTI_NOP; } }
void reference_g(const std::uint32_t n) { for (std::uint32_t i = 0; i < n; i++) { TTI_NOP; } }
void g(const std::uint32_t n)           { for (std::uint32_t i = 1; i <= n; i++) { TTI_NOP; } }
void reference_h(const std::uint32_t n) { for (std::uint32_t i = 0; i < n; i++) { TTI_NOP; } }
void h(const std::uint32_t n)           { std::uint32_t k = n; while (k--) { TTI_NOP; } }
void reference_k(const std::uint32_t n) { for (std::uint32_t i = 0; i < n; i++) { TTI_NOP; } }
void k(const std::uint32_t m)           { for (std::uint32_t i = 0; i < m; i++) { TTI_NOP; } }
void reference_p(const std::uint32_t n) { for (std::uint32_t i = 0; i < n; i++) { TTI_NOP; } }
void p(const std::uint32_t n)           { for (std::uint32_t i = 0; i < n; i += 2) { TTI_NOP; } }
