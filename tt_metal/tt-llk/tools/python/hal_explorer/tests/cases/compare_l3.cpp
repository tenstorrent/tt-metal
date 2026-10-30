// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_f(const std::uint32_t n) { std::uint32_t v = 1; for (std::uint32_t i = 0; i < n; i++) { TT_SETC16(5, v); v = v * 3; } }
void f(const std::uint32_t n)           { for (std::uint32_t i = 0; i < n; i++) { TT_SETC16(5, 1); } }
