// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_f(const std::uint32_t x) { switch (x) { case 0: TTI_STALLWAIT(1,2); break; case 1: TTI_STALLWAIT(2,2); break; case 2: TTI_STALLWAIT(3,2); break; case 3: TTI_STALLWAIT(4,2); break; case 4: TTI_STALLWAIT(5,2); break; case 5: TTI_STALLWAIT(6,2); break; case 6: TTI_STALLWAIT(7,2); break; case 7: TTI_STALLWAIT(8,2); break; default: break; } }
void f(const std::uint32_t x) { switch (x) { case 0: TTI_STALLWAIT(1,2); break; case 1: TTI_STALLWAIT(2,2); break; case 2: TTI_STALLWAIT(3,2); break; case 3: TTI_STALLWAIT(4,2); break; case 4: TTI_STALLWAIT(5,2); break; case 5: TTI_STALLWAIT(6,2); break; case 6: TTI_STALLWAIT(7,2); break; case 7: TTI_STALLWAIT(9,2); break; default: break; } }
void reference_g(const std::uint32_t x) { switch (x) { case 0: TTI_STALLWAIT(1,2); break; case 1: TTI_STALLWAIT(2,2); break; case 2: TTI_STALLWAIT(3,2); break; case 3: TTI_STALLWAIT(4,2); break; case 4: TTI_STALLWAIT(5,2); break; case 5: TTI_STALLWAIT(6,2); break; case 6: TTI_STALLWAIT(7,2); break; case 7: TTI_STALLWAIT(8,2); break; default: break; } }
void g(const std::uint32_t x) { if (x == 0) { TTI_STALLWAIT(1,2); } else if (x == 1) { TTI_STALLWAIT(2,2); } else if (x == 2) { TTI_STALLWAIT(3,2); } else if (x == 3) { TTI_STALLWAIT(4,2); } else if (x == 4) { TTI_STALLWAIT(5,2); } else if (x == 5) { TTI_STALLWAIT(6,2); } else if (x == 6) { TTI_STALLWAIT(7,2); } else if (x == 7) { TTI_STALLWAIT(8,2); } }
