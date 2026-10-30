// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// identical runtime expressions that GCC prints with SSA names (ternary, min, unary, division)
void reference_t(const std::uint32_t a, const std::uint32_t b) { TT_SETC16(1, a > b ? a : b); }
void t(const std::uint32_t a, const std::uint32_t b) { TT_SETC16(1, a > b ? a : b); }
void reference_u(const std::uint32_t a, const std::uint32_t b) { TT_SETC16(1, a < b ? a : b); }
void u(const std::uint32_t a, const std::uint32_t b) { TT_SETC16(1, b < a ? b : a); }
void reference_v(const std::uint32_t a) { TT_SETC16(1, ~a); }
void v(const std::uint32_t a) { TT_SETC16(1, ~a); }
void reference_w(const std::uint32_t a) { TT_SETC16(1, a / 3); }
void w(const std::uint32_t a) { TT_SETC16(1, a / 3); }
void reference_x(const std::uint32_t a) { TT_SETC16(1, a % 3); }
void x(const std::uint32_t a) { TT_SETC16(1, a % 3); }
void reference_y(const std::uint32_t a, const std::uint32_t b) { TT_SETC16(1, a / b); }
void y(const std::uint32_t a, const std::uint32_t b) { TT_SETC16(1, a / b); }
