// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Loop-carried induction value used in an instruction word.
void reference_f(const std::uint32_t n) { for (std::uint32_t i = 0; i < n; i++) { TT_SETC16(5, i); } }
void f(const std::uint32_t n) { for (std::uint32_t i = 0; i < n; i++) { TT_SETC16(5, 2 * i); } }
void reference_g(const std::uint32_t n) { for (std::uint32_t i = 0; i < n; i++) { TT_SETC16(5, i); } }
void g(const std::uint32_t n) { for (std::uint32_t i = 0; i < n; i++) { TT_SETC16(5, 0); } }
