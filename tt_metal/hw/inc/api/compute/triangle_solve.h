// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "api/dataflow/circular_buffer.h"
#ifdef TRISC_MATH
#include "llk_math_triangle_solve_sfpu_entry.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Solves L X = RHS for one 32x32 tile by forward substitution on the SFPU, where L is unit lower-triangular
 * (the diagonal is implicit and never read). L is read in place from L1, tile l_tile_idx of cb_l, which must be
 * resident (*cb_wait_front* done) when the call is made; on return the unpack thread has waited for the math
 * thread to finish reading it, so the tile may be popped or overwritten right after this call. Only the
 * right-hand side occupies a DST register. The solution is left in DST register idst_out in standard tile
 * layout. Requires a prior call to *triangle_solve_tile_init*. The DST register buffer must be in the acquired
 * state via *tile_regs_acquire*, and the kernel must run with 32-bit destination accumulation. Blackhole only;
 * on other architectures the call asserts. This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument            | Description                                                                          | Type             | Valid Range                                          | Required |
 * |---------------------|--------------------------------------------------------------------------------------|------------------|------------------------------------------------------|----------|
 * | L_FORMAT            | Data format of the L tile in L1                                                      | DataFormat       | Float32 (default) or Float16_b                       | False    |
 * | L_NEGATED           | The tile holds -L below the diagonal (L's strict-lower entries supplied negated)     | bool             | true or false (default)                              | False    |
 * | is_fp32_dest_acc_en | 32-bit destination accumulation is enabled                                           | bool             | true (default: the kernel's fp32_dest_acc_en)        | False    |
 * | cb_l                | Circular buffer holding the L tile                                                   | CircularBuffer&  | A CB whose front tiles are in L_FORMAT               | True     |
 * | l_tile_idx          | Index of the L tile within cb_l, relative to its front                               | uint32_t         | Less than the number of front-waited tiles           | True     |
 * | idst_in             | DST register index of the right-hand side                                            | uint32_t         | Must be less than the size of the DST register buffer | True    |
 * | idst_out            | DST register index that receives the solution X                                      | uint32_t         | Must be less than the size of the DST register buffer, not idst_in | True |
 */
// clang-format on
template <DataFormat L_FORMAT = DataFormat::Float32, bool L_NEGATED = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void triangle_solve_tile(
    [[maybe_unused]] CircularBuffer& cb_l,
    [[maybe_unused]] uint32_t l_tile_idx,
    [[maybe_unused]] uint32_t idst_in,
    [[maybe_unused]] uint32_t idst_out) {
    static_assert(is_fp32_dest_acc_en, "triangle_solve_tile needs 32-bit destination accumulation (fp32_dest_acc_en)");
    // UNPACK resolves the tile's L1 address and mailboxes it to MATH and PACK.
    const uint32_t l1_base = cb_l.get_tile_address(l_tile_idx);
    MATH((llk_math_triangle_solve_sfpu_tile<L_FORMAT, L_NEGATED>(l1_base, idst_in, idst_out)));
    // MATH reads L from L1 itself, which nothing orders against UNPACK's later cb_pop_front of cb_l: MATH signals
    // when its reads are done and UNPACK waits for that before returning.
    MATH((mailbox_write(ckernel::ThreadId::UnpackThreadId, 1)));
    UNPACK((mailbox_read(ckernel::ThreadId::MathThreadId)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void triangle_solve_tile_init() { MATH((llk_math_triangle_solve_sfpu_init())); }

}  // namespace ckernel
