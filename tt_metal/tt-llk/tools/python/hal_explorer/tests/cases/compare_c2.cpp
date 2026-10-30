// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// equivalent conditions, different spelling
void reference_or(const std::uint32_t a, const std::uint32_t b) { if (a || b) { TTI_NOP; } }
void or_(const std::uint32_t a, const std::uint32_t b) { if (a) { TTI_NOP; } else if (b) { TTI_NOP; } }

void reference_and(const std::uint32_t a, const std::uint32_t b) { if (a && b) { TTI_NOP; } }
void and_(const std::uint32_t a, const std::uint32_t b) { if (a) { if (b) { TTI_NOP; } } }

void reference_not(const std::uint32_t a, const std::uint32_t b) { if (!(a && b)) { TTI_NOP; } else { TTI_STALLWAIT(1,2); } }
void not_(const std::uint32_t a, const std::uint32_t b) { if (!a || !b) { TTI_NOP; } else { TTI_STALLWAIT(1,2); } }

void reference_rng(const std::uint32_t x) { if (x == 1 || x == 2) { TTI_NOP; } }
void rng(const std::uint32_t x) { if (x >= 1 && x <= 2) { TTI_NOP; } }

void reference_sw(const std::uint32_t x) { switch (x) { case 1: TTI_NOP; break; case 2: TTI_STALLWAIT(1,2); break; default: break; } }
void sw(const std::uint32_t x) { if (x == 1) { TTI_NOP; } else if (x == 2) { TTI_STALLWAIT(1,2); } }

void reference_gt(const std::uint32_t x) { if (x != 0) { TTI_NOP; } }
void gt(const std::uint32_t x) { if (x > 0) { TTI_NOP; } }

void reference_ge(const std::uint32_t x) { if (x >= 4) { TTI_NOP; } }
void ge(const std::uint32_t x) { if (x > 3) { TTI_NOP; } }

void reference_bo(const bool a, const bool b) { if (a && b) { TTI_NOP; } }
void bo(const bool a, const bool b) { if (a) { if (b) { TTI_NOP; } } }
