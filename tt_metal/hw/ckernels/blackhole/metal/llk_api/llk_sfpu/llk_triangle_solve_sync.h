// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_assert.h"

namespace ckernel {

// MATH -> UNPACK handshake of triangle_solve_tile.
// MATH: during _triangle_solve_tile_ the MATH RISC loads the strict-lower entries of the L tile from L1 with its own
// loads (no unpacker involved). UNPACK: after triangle_solve_tile returns, the caller's cb_pop_front of cb_l executes
// on the UNPACK thread and frees the slot for the next tile. Problem: the CB does not synchronize the two threads, so
// the pop and a refill of the slot can precede the MATH RISC's last load and overwrite L during the solve. Solution:
// once its last load has returned, the MATH RISC posts one release per solve on the MATH -> UNPACK mailbox, and
// triangle_solve_tile does not return on the UNPACK thread until that release has been consumed. The mailbox FIFO is
// shared with other handshakes and pairs by program order only; the tag lets the wait assert on a mispaired message.
constexpr std::uint32_t TRIANGLE_SOLVE_L_RELEASED = 0x54534C52;  // 'TSLR'

/**
 * @brief Release the L tile to the UNPACK thread: post the tagged message on the MATH -> UNPACK mailbox.
 *
 * Called by @ref llk_math_triangle_solve_sfpu_tile once every L1 load of L has returned.
 *
 * @note Pair every call with one @ref llk_unpack_triangle_solve_wait_l_released on the UNPACK thread, in the same
 *       program order as other users of the MATH -> UNPACK mailbox.
 */
inline void llk_math_triangle_solve_release_l() { mailbox_write(ThreadId::UnpackThreadId, TRIANGLE_SOLVE_L_RELEASED); }

/**
 * @brief Wait on the UNPACK thread until MATH has released the L tile: block on the MATH -> UNPACK mailbox and check
 * the tag.
 *
 * Returns once the message posted by @ref llk_math_triangle_solve_release_l has been consumed; the L tile may be popped
 * or overwritten afterwards.
 *
 * @note Call after the MATH thread's @ref llk_math_triangle_solve_sfpu_tile and before the cb_pop_front of the L tile's
 *       CB. A message other than the release tag trips LLK_ASSERT (live with TT_METAL_LLK_ASSERTS=1).
 */
inline void llk_unpack_triangle_solve_wait_l_released() {
    [[maybe_unused]] const std::uint32_t msg = mailbox_read(ThreadId::MathThreadId);
    LLK_ASSERT(msg == TRIANGLE_SOLVE_L_RELEASED, "triangle_solve_tile: unexpected MATH -> UNPACK mailbox message");
}

}  // namespace ckernel
