// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This file is (also) used for linker script-generation, so cannot
// contain non-cpp items outside of __cplusplus regions.
// Linker-only information is inside !__cplusplus regions (we don't
// want that accidentally leaking into C++).

// Writes here are appended to the tensix core instruction FIFO. This
// has priority over incoming instruction fetch returns, which are
// simply dropped. The instruction will stay in the queue if a loop
// instruction is in progress. If the FIFO gets overfull, writes are
// dropped? Additionally, the instruction queue is flushed in some
// cases.

#if !defined(__cplusplus)
#define INSTRN_BUF_BASE 0xFFE40000
#endif
// The HAL needs to know the size-per cpu
#define INSTRN_BUF_STRIDE 0x00010000
