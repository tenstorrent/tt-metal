// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Hardware register map for the DRISC-visible Blackhole GDDR Memory Controller (MC) — do not include directly.

#ifdef COMPILE_FOR_DRISC

// GDDR Memory Controller Multi-Port Front End (MPFE) priority fields.
// The MC register names P1/P2/P3 correspond to the Blackhole DRAM tile names D0/D1/D2.
#define GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_P1_REG_ADDR (0xFC105830u)
#define GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_P2_REG_ADDR (0xFC105834u)
#define GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_P3_REG_ADDR (0xFC105838u)

#define GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_MASK (0x7u)
#define GDDR_MC_MPFE_CFG_ROUNDROBIN_WEIGHT_DEFAULT (0x0u)

#endif  // COMPILE_FOR_DRISC
