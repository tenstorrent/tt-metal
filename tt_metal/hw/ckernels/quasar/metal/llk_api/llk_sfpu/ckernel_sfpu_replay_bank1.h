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

// Replay bank 1 holds SFPU bodies (cumsum, ema) clear of the bank-0 FPU recordings; a REPLAY with
// `last` flips the bank. Its users overwrite each other, so switching needs the other's init again.
constexpr std::uint32_t SFPU_REPLAY_BANK_DEPTH = 32;

// The read ID only flips after an executed REPLAY, so entering bank 1 replays an SFPNOP kept here.
constexpr std::uint32_t SFPU_REPLAY_BANK_SWITCH_SLOT = SFPU_REPLAY_BANK_DEPTH - 1;

// On quasar_4row, 16-bit transpose_dest's recording ends in the switch slot; re-init it after
// bank-1 SFPU tiles there. On the 8-row FPU every bank-0 recording stops by slot 23.
constexpr bool SFPU_REPLAY_BANK_SWITCH_SLOT_FREE = (ELTWISE_MATH_ROWS == 8);

/**
 * @brief Record LEN instructions from @p body into replay bank 1 at START, without executing them.
 */
template <std::uint32_t START, std::uint32_t LEN, typename F>
inline void _sfpu_record_replay_bank1_(F body) {
    static_assert(START + LEN <= SFPU_REPLAY_BANK_DEPTH, "the recorded body must fit one replay bank");

    // `last` flips the write ID to bank 1
    TTI_REPLAY(
        SFPU_REPLAY_BANK_SWITCH_SLOT, 1 /*len*/, 1 /*last*/, 0 /*set_mutex*/, 0 /*exec_while_loading*/, 1 /*load*/);
    TTI_SFPNOP(0 /*srcs_wr_done*/, 0 /*srcs_rd_done*/, 0 /*dest_done*/);

    // `last` flips it back to bank 0
    load_replay_buf<START, LEN, false /*exec_while_loading*/, 0 /*set_mutex*/, 1 /*last*/>(body);
}

/**
 * @brief Point the read ID at bank 1; the caller's last bank-1 REPLAY must set `last`.
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
