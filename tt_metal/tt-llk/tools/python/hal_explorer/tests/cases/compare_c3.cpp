// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_a1(const std::uint32_t a, const std::uint32_t b) { if (a || b) { TTI_NOP; } }
void a1(const std::uint32_t a, const std::uint32_t b) { if (a) { TTI_NOP; } else if (b) { TTI_NOP; } }

void reference_a2(const std::uint32_t a, const std::uint32_t b) { if (a && b) { TTI_NOP; } }
void a2(const std::uint32_t a, const std::uint32_t b) { if (a) { if (b) { TTI_NOP; } } }

void reference_a3(const std::uint32_t a, const std::uint32_t b) { if (!(a && b)) { TTI_NOP; } else { TTI_STALLWAIT(1,2); } }
void a3(const std::uint32_t a, const std::uint32_t b) { if (!a || !b) { TTI_NOP; } else { TTI_STALLWAIT(1,2); } }

void reference_a4(const std::uint32_t x) { if (x == 1 || x == 2) { TTI_NOP; } }
void a4(const std::uint32_t x) { if (x >= 1 && x <= 2) { TTI_NOP; } }

void reference_a5(const std::uint32_t a, const std::uint32_t b) { if (a) { TTI_NOP; } if (b) { TTI_STALLWAIT(1,2); } }
void a5(const std::uint32_t a, const std::uint32_t b) { if (b) { TTI_STALLWAIT(1,2); } if (a) { TTI_NOP; } }
