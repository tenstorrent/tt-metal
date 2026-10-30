// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tool: both paths count 2.  Binary: c => ebreak, 0 instructions issued.
void trap_path(const bool c) { if (c) { __builtin_trap(); } TTI_NOP; TTI_NOP; }
