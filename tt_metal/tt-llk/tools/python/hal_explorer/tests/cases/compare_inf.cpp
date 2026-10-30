// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_f(const std::uint32_t x) { if (x > 5) { TTI_NOP; } if (x < 3) { TTI_STALLWAIT(1,2); } }
void f(const std::uint32_t x) { if (x > 5) { TTI_NOP; } else if (x < 3) { TTI_STALLWAIT(1,2); } }
void reference_g(const std::uint32_t x) { if (x == 1) { TTI_NOP; } if (x == 2) { TTI_STALLWAIT(1,2); } }
void g(const std::uint32_t x) { if (x == 1) { TTI_NOP; } else if (x == 2) { TTI_STALLWAIT(1,2); } }
void reference_h(const std::uint32_t x, const std::uint32_t y) { if (x) { TTI_NOP; } if (y) { TTI_STALLWAIT(1,2); } }
void h(const std::uint32_t x, const std::uint32_t y) { if (x) { TTI_NOP; if (y) { TTI_STALLWAIT(1,2); } } else if (y) { TTI_STALLWAIT(1,2); } }
