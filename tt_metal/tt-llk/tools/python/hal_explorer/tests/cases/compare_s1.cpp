// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_a(const std::int32_t x) { TT_SETC16(1, x >> 3); }
void a(const std::uint32_t x) { TT_SETC16(1, x >> 3); }
void reference_b(const std::int32_t x) { if (x < 5) { TTI_NOP; } }
void b(const std::uint32_t x) { if (x < 5) { TTI_NOP; } }
void reference_c(const std::int32_t x) { TT_SETC16(1, x / 4); }
void c(const std::uint32_t x) { TT_SETC16(1, x / 4); }
void reference_d(const std::int32_t x) { for (std::int32_t i = 0; i < x; i++) { TTI_NOP; } }
void d(const std::uint32_t x) { for (std::uint32_t i = 0; i < x; i++) { TTI_NOP; } }
