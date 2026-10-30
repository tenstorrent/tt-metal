// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_sw(const std::uint32_t x) { switch (x) { case 1: TTI_NOP; break; case 2: TTI_STALLWAIT(1,2); break; default: break; } }
void sw(const std::uint32_t x) { if (x == 1) { TTI_NOP; } else if (x == 2) { TTI_STALLWAIT(1,2); } }
