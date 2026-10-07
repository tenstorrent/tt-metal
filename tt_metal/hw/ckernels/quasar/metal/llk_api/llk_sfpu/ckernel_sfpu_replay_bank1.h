// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_ops.h"
#include "cmath_common.h"

namespace ckernel {
namespace sfpu {

// Replay bank 1 for SFPU bodies that must survive math-thread FPU ops (cumsum, ema).
//
// The math thread's replay buffer is double-banked: 64 entries, two banks of 32. A REPLAY's start
// index addresses within a bank, and which bank it hits comes from a write ID (loads) and a read ID
// (executes), each flipped by a REPLAY with `last` set once it completes. Math-thread FPU ops record
// into and replay from bank 0 with both IDs at 0, so a body kept in bank 1 survives them and an FPU
// op can run between that SFPU op's tiles without a re-init. Bank 1 itself holds one body at a time:
// its users (cumsum, ema) overwrite each other's recording, so switching between them needs the
// other's init again. Both IDs are back at 0 whenever none of them is running.
constexpr std::uint32_t SFPU_REPLAY_BANK_DEPTH = 32;

// The read ID only flips after an executed REPLAY, so entering bank 1 costs one instruction replayed
// out of bank 0: an SFPNOP kept in bank 0's last slot, refreshed on every entry.
constexpr std::uint32_t SFPU_REPLAY_BANK_SWITCH_SLOT = SFPU_REPLAY_BANK_DEPTH - 1;

// Whether every math-thread FPU recording stops short of the bank-switch slot, so refreshing it
// clobbers no FPU op's resident recording. The longest is transpose_dest's transpose-of-faces body:
// on the 8-row FPU 16 entries (16-bit Dest) or 24 (32-bit), clear of slot 31. On quasar_4row the
// 16-bit body is 32 entries and ends in slot 31, so there a transpose_dest interleaved with
// bank-1 SFPU tiles needs its init again after them before it runs.
constexpr bool SFPU_REPLAY_BANK_SWITCH_SLOT_FREE = (ELTWISE_MATH_ROWS == 8);

/**
 * @brief Record a body into replay bank 1 without executing it, leaving both bank IDs at 0.
 *
 * @tparam START: First bank-1 slot of the recording.
 * @tparam LEN: Number of instructions @p body issues.
 * @param body: Callable issuing exactly LEN immediate instructions.
 */
template <std::uint32_t START, std::uint32_t LEN, typename F>
inline void _sfpu_record_replay_bank1_(F body) {
    static_assert(START + LEN <= SFPU_REPLAY_BANK_DEPTH, "the recorded body must fit one replay bank");

    // Loading the bank switch with `last` flips the write ID to bank 1 for the body below.
    TTI_REPLAY(
        SFPU_REPLAY_BANK_SWITCH_SLOT, 1 /*len*/, 1 /*last*/, 0 /*set_mutex*/, 0 /*exec_while_loading*/, 1 /*load*/);
    TTI_SFPNOP(0 /*srcs_wr_done*/, 0 /*srcs_rd_done*/, 0 /*dest_done*/);

    // Record only; `last` flips the write ID back to bank 0 for everyone else.
    load_replay_buf<START, LEN, false /*exec_while_loading*/, 0 /*set_mutex*/, 1 /*last*/>(body);
}

/**
 * @brief Point the read ID at replay bank 1, so the following REPLAYs run bank-1 recordings.
 *
 * Refreshes the SFPNOP in bank 0's last slot and replays it with `last`. The caller's final REPLAY
 * out of bank 1 must set `last` to flip the read ID back to bank 0.
 *
 * @note Overwrites bank 0 slot SFPU_REPLAY_BANK_SWITCH_SLOT; see SFPU_REPLAY_BANK_SWITCH_SLOT_FREE.
 */
inline void _sfpu_enter_replay_bank1_() {
    TTI_REPLAY(
        SFPU_REPLAY_BANK_SWITCH_SLOT, 1 /*len*/, 0 /*last*/, 0 /*set_mutex*/, 0 /*exec_while_loading*/, 1 /*load*/);
    TTI_SFPNOP(0 /*srcs_wr_done*/, 0 /*srcs_rd_done*/, 0 /*dest_done*/);
    TTI_REPLAY(
        SFPU_REPLAY_BANK_SWITCH_SLOT, 1 /*len*/, 1 /*last*/, 0 /*set_mutex*/, 0 /*exec_while_loading*/, 0 /*load*/);
}

}  // namespace sfpu
}  // namespace ckernel
