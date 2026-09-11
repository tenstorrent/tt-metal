// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Hardware register map for the DRISC-visible Blackhole GDDR memory controller
// priority fields — do not include directly.

#ifdef COMPILE_FOR_DRISC

#include <stdint.h>

#define GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_P1_REG_ADDR (0xFC105830u)
#define GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_P2_REG_ADDR (0xFC105834u)
#define GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_P3_REG_ADDR (0xFC105838u)

#define GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_MASK (0x7u)
#define GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_DEFAULT (0x0u)

static inline uint32_t gddr_mc_mpfe_weight_reg_addr(uint32_t port) {
    return GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_P1_REG_ADDR + (port - 1u) * sizeof(uint32_t);
}

static inline uint32_t gddr_mc_read_mpfe_weight(uint32_t port) {
    return *reinterpret_cast<volatile uint32_t*>(gddr_mc_mpfe_weight_reg_addr(port)) &
           GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_MASK;
}

static inline void gddr_mc_write_mpfe_weight(uint32_t port, uint32_t weight) {
    volatile uint32_t* reg = reinterpret_cast<volatile uint32_t*>(gddr_mc_mpfe_weight_reg_addr(port));
    const uint32_t current = *reg;
    *reg = (current & ~GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_MASK) | (weight & GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_MASK);
}

#endif  // COMPILE_FOR_DRISC
