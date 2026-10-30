// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// same names, permuted positions: callers pass different values
void reference_f(const std::uint32_t a, const std::uint32_t b) { if (a) { TTI_NOP; } }
void f(const std::uint32_t b, const std::uint32_t a) { if (a) { TTI_NOP; } }
// different parameter count
void reference_g(const std::uint32_t a) { if (a) { TTI_NOP; } }
void g(const std::uint32_t a, const std::uint32_t b) { if (a) { TTI_NOP; } }
