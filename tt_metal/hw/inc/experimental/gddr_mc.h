// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "api/debug/assert.h"
#include "internal/tt-1xx/blackhole/gddr_mc_regs.h"

#ifdef COMPILE_FOR_DRISC

constexpr uint32_t gddr_mc_mpfe_weight_reg_addr(uint32_t port) {
    switch (port) {
        case 1: return GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_P1_REG_ADDR;
        case 2: return GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_P2_REG_ADDR;
        case 3: return GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_P3_REG_ADDR;
        default: ASSERT(false, DebugAssertTripped); __builtin_trap();
    }
}

inline uint32_t gddr_mc_read_mpfe_weight(uint32_t port) {
    return *reinterpret_cast<volatile uint32_t*>(gddr_mc_mpfe_weight_reg_addr(port)) &
           GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_MASK;
}

inline void gddr_mc_write_mpfe_weight(uint32_t port, uint32_t weight) {
    volatile uint32_t* reg = reinterpret_cast<volatile uint32_t*>(gddr_mc_mpfe_weight_reg_addr(port));
    const uint32_t current = *reg;
    *reg = (current & ~GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_MASK) | (weight & GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_MASK);
}

#endif  // COMPILE_FOR_DRISC
