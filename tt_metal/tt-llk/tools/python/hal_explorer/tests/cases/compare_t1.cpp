// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_f(const std::uint32_t b0, const std::uint32_t b1, const std::uint32_t b2, const std::uint32_t b3, const std::uint32_t b4, const std::uint32_t b5, const std::uint32_t b6, const std::uint32_t b7, const std::uint32_t b8, const std::uint32_t b9, const std::uint32_t b10) { if (b0) { TTI_NOP; }; if (b1) { TTI_NOP; }; if (b2) { TTI_NOP; }; if (b3) { TTI_NOP; }; if (b4) { TTI_NOP; }; if (b5) { TTI_NOP; }; if (b6) { TTI_NOP; }; if (b7) { TTI_NOP; }; if (b8) { TTI_NOP; }; if (b9) { TTI_NOP; }; if (b10) { TTI_NOP; }; }
void f(const std::uint32_t b0, const std::uint32_t b1, const std::uint32_t b2, const std::uint32_t b3, const std::uint32_t b4, const std::uint32_t b5, const std::uint32_t b6, const std::uint32_t b7, const std::uint32_t b8, const std::uint32_t b9, const std::uint32_t b10) { if (!b0) { TTI_STALLWAIT(1,2); } if (b0) { TTI_NOP; }; if (b1) { TTI_NOP; }; if (b2) { TTI_NOP; }; if (b3) { TTI_NOP; }; if (b4) { TTI_NOP; }; if (b5) { TTI_NOP; }; if (b6) { TTI_NOP; }; if (b7) { TTI_NOP; }; if (b8) { TTI_NOP; }; if (b9) { TTI_NOP; }; if (b10) { TTI_NOP; }; }
