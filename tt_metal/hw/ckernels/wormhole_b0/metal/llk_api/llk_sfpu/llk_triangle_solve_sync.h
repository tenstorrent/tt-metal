// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel {

// The SFPU triangle solve is not implemented on Wormhole: the MATH-side entry rejects a call at compile time, so the
// UNPACK side of its MATH -> UNPACK handshake has nothing to wait for.
inline void llk_unpack_triangle_solve_wait_l_released() {}

}  // namespace ckernel
