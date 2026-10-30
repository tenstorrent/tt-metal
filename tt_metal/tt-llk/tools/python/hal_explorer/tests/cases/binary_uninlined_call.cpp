// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tool: count 1, no warning.  Binary: 1 + 3 (tail call into helper_issue).
__attribute__((noinline)) void helper_issue() { TTI_NOP; TTI_NOP; TTI_NOP; }
void calls_helper() { TTI_NOP; helper_issue(); }
