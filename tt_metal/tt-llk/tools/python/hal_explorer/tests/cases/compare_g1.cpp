// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// reference is else-if (exclusive); rewrite issues both when a && b
void reference_f(const std::uint32_t a, const std::uint32_t b) { if (a) { TTI_NOP; } else if (b) { TTI_STALLWAIT(1,2); } }
void f(const std::uint32_t a, const std::uint32_t b) { if (a) { TTI_NOP; } if (b) { TTI_STALLWAIT(1,2); } }
