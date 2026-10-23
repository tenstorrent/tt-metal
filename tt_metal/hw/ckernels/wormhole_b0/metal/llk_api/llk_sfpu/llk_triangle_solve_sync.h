// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel {

// The SFPU triangle solve is not implemented on Wormhole.

/**
 * @brief No-op: the MATH-side entry rejects triangle_solve_tile at compile time on this architecture, so there is no
 * release to wait for.
 */
inline void llk_unpack_triangle_solve_wait_l_released() {}

}  // namespace ckernel
