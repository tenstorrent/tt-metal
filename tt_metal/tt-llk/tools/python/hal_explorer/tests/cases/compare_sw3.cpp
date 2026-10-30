// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_sw(const std::uint32_t x) {
  switch (x) { case 0: TTI_NOP; break; case 1: TTI_STALLWAIT(1,2); break; case 2: TTI_NOP; TTI_NOP; break; case 3: TTI_STALLWAIT(2,4); break; case 4: TTI_STALLWAIT(2,6); break;case 5: TTI_STALLWAIT(3,6); break; case 6: TTI_STALLWAIT(5,6); break; case 7: TTI_STALLWAIT(5,7); break; default: break; } }
