// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_assert.h"

namespace ckernel {

// MATH -> UNPACK handshake of triangle_solve_tile. The MATH RISC reads the L tile straight from L1, which nothing
// orders against UNPACK's later cb_pop_front of that CB; MATH releases the tile once its reads have returned and
// UNPACK waits for the release before the API returns. The message travels on the MATH -> UNPACK mailbox FIFO, which
// other handshakes share (e.g. the fp32 dest-acc reconfig), and pairs up with this wait only by program order; the
// tag makes a mismatch assert in an asserting build instead of being consumed as another handshake's payload.
constexpr std::uint32_t TRIANGLE_SOLVE_L_RELEASED = 0x54534C52;  // 'TSLR'

inline void llk_math_triangle_solve_release_l() { mailbox_write(ThreadId::UnpackThreadId, TRIANGLE_SOLVE_L_RELEASED); }

inline void llk_unpack_triangle_solve_wait_l_released() {
    [[maybe_unused]] const std::uint32_t msg = mailbox_read(ThreadId::MathThreadId);
    LLK_ASSERT(msg == TRIANGLE_SOLVE_L_RELEASED, "triangle_solve_tile: unexpected MATH -> UNPACK mailbox message");
}

}  // namespace ckernel
