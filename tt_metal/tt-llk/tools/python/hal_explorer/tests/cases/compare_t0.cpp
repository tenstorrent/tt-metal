// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_f(const std::uint32_t a) { TTI_NOP; if (a) { TTI_STALLWAIT(1,2); } }
void f(const std::uint32_t b) { TTI_NOP; if (b) { TTI_STALLWAIT(1,2); } }
