// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// TTI REPLAY LOWERING FRAMEWORK — shared scaffolding for every kernel body
// that opts into the Tensix replay-buffer lowering (piecewise_generic.cpp
// poly/exp/log/trig shapes AND piecewise_rational.cpp rational shapes).
//
// Any eval kind can opt in by providing (a) a constexpr body-slot count, (b) a
// hand-scheduled TTI body obeying the hazard rules below, (c) a per-tile
// sequence: [SFPLOADI constant pins] -> TT_REPLAY_RECORD(slots) -> body ->
// TT_REPLAY_REST(slots, n-1) -> TT_REPLAY_TILE_EPILOGUE(). The opt-in gate
// MUST be a constexpr `k<Kind>TtiReplay` that folds capacity + shape checks;
// when it is false the kind's existing sfpi loop compiles unchanged (clean
// fallback — the two paths coexist behind `if constexpr`).
//
// HAZARD RULES (silicon-derived; NO offline check catches violations):
//   1. A MAD-unit result (SFPMAD/SFPADD/SFPMUL/SFPADDI/SFPMULI/SFPLUT*) is NOT
//      readable by the immediately-following instruction at replay stream
//      rate. Every consumer must sit >= 1 instruction after its producer.
//      Bit/int-unit results (SFPEXEXP/SFPEXMAN/SFPSHFT/SFPSETEXP/SFPSETSGN/
//      SFPSETMAN/SFPIADD/SFPABS/SFPCAST/SFP_STOCH_RND/SFPLOADI) are readable
//      gap-0. SFPLOAD is treated as needing a gap (no counter-evidence).
//      2-element interleave provides gaps for free; 1-element bodies use the
//      independent side-computation ops as fillers, then SFPNOP.
//   2. The body's dst walk (ADDR_MOD_6 dest+= on the final SFPSTORE) displaces
//      the D-RWC counter, which the next tile's A2D datacopy MOP shares. The
//      per-tile sequence MUST end with TT_REPLAY_TILE_EPILOGUE() (SETRWC SET_D)
//      or subsequent tiles copy 64 rows off.
//   3. Replay buffer capacity is 32 slots (start_idx 0 + len <= 32).
//      Instructions past slot 31 execute once at record time and silently
//      vanish from replays. Gate on tti_replay_fits(); prefer <= 16 slots
//      (slots 16-31 are conventionally FPU-side).
//
// Predication (CC) in replay bodies is functionally sanctioned IF (a) the
// body is CC-balanced (SETCC ... ENCC fully inside the recorded body) and
// (b) any SETCC reading a MAD-unit result gets the rule-1 hazard gap.
//
// Register conventions (BH): L0-L7 general (bodies own them inside the tile
// loop), L8 = 0.8373, L9 = 0.0, L10 = 1.0, L11 = -1.0, L12/13/14 =
// vConstFloatPrgm0/1/2 (programmed once in the pre-tile-loop preamble).
// ============================================================================

#pragma once

#include <cstdint>
