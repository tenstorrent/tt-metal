// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_sw(const std::uint32_t x) {
  switch (x) { case 1: TTI_NOP; break; case 5: TTI_STALLWAIT(1,2); break; case 9: TTI_NOP; TTI_NOP; break; case 20: TTI_STALLWAIT(2,4); break; default: break; } }
void sw(const std::uint32_t x) {
  switch (x) { case 1: TTI_NOP; break; case 5: TTI_STALLWAIT(1,2); break; case 9: TTI_NOP; TTI_NOP; break; case 20: TTI_STALLWAIT(2,5); break; default: break; } }
