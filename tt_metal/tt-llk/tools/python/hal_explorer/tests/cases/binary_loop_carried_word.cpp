// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Count right (80 / 40*2), words wrong: loop body words are replicated from iteration 0.
// Real ADDDMAREG words: 0x58801001, 0x58801041, 0x58801081, ...
void unrolled_by_tool() { for (std::uint32_t i = 0; i < 40; ++i) { scalar::add(hal::gpr<1>(), hal::gpr<1>(), scalar::immediate(i)); TTI_NOP; } }
// Runtime-trip loop: body printed as constant 0x58801001; repeat text "while n != (0 + 1)" instead of "n".
void rt_loop(const std::uint32_t n) { for (std::uint32_t i = 0; i < n; ++i) { scalar::add(hal::gpr<1>(), hal::gpr<1>(), scalar::immediate(i)); } }
