// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel_sfpu_softsign_bw_bf16_replay_window.h"

#define TT_REPLAY_ADDRMOD_EXIT() ((void)0)

// Replay operands: section 7 holds the dest address and section 6 advances it.
#define TT_ADDRMOD_DST_HOLD ckernel::ADDR_MOD_7

#define TT_REPLAY_TILE_EPILOGUE()                                                      \
    do {                                                                               \
        TTI_SETRWC(ckernel::p_setrwc::CLR_NONE, 0, 0, 0, 0, ckernel::p_setrwc::SET_D); \
        TT_REPLAY_ADDRMOD_EXIT();                                                      \
    } while (0)
// Pin a full fp32 constant into a general LREG (2 issue slots, per tile).
