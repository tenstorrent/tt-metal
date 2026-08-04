// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Expert index tensor encoding shared by matmul_expert SRAM/DRAM kernels and op.py.
// Bit 15 distinguishes SRAM (1) from DRAM (0) routing; bits 0..14 carry either the
// compact SRAM slot index or the global DRAM expert id.
namespace deepseek_b1_ops {

inline constexpr uint32_t EXPERT_SRAM_FLAG = 0x8000;
inline constexpr uint32_t EXPERT_SLOT_MASK = 0x7FFF;

FORCE_INLINE bool is_sram_expert(uint32_t raw_idx) { return (raw_idx & EXPERT_SRAM_FLAG) != 0; }
FORCE_INLINE uint32_t expert_slot(uint32_t raw_idx) { return raw_idx & EXPERT_SLOT_MASK; }

}  // namespace deepseek_b1_ops
