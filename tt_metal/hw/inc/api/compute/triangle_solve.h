// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/cb_api.h"
#include "api/compute/common_globals.h"
#include "llk_assert.h"
#ifdef TRISC_MATH
#include "llk_math_triangle_solve_sfpu_entry.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Solves L X = RHS for one 32x32 tile by forward substitution on the SFPU, where L is unit lower-triangular
 * (the diagonal is implicit and never read). L is read in place from L1, tile l_tile_idx of cb_l, and never
 * occupies DST; RHS and X use two distinct DST tiles, and X is left in idst_out in standard tile layout. cb_l
 * must be resident (*cb_wait_front* done) when the call is made; on return the unpack thread has waited for
 * the math thread to finish reading it, so the tile may be popped or overwritten right after this call.
 * Requires a prior call to *triangle_solve_tile_init*. The DST register buffer must be in the acquired state
 * via *tile_regs_acquire*, and the kernel must run with 32-bit destination accumulation. Blackhole only: on
 * other architectures a call does not compile. This call is blocking and is only available on the compute
 * engine.
 *
 * Return value: None
 *
 * | Param Type | Name                | Description                                                     | Type            | Valid Range                                                | Required |
 * |------------|---------------------|-----------------------------------------------------------------|-----------------|------------------------------------------------------------|----------|
 * | Template   | L_FORMAT            | Data format of the L tile in L1; must be cb_l's data format     | DataFormat      | Float32 (default), Float16_b                               | False    |
 * | Template   | L_NEGATED           | The tile holds -L below the diagonal                            | bool            | true, false (default)                                      | False    |
 * | Template   | is_fp32_dest_acc_en | 32-bit destination accumulation is enabled                      | bool            | true (default: the kernel's fp32_dest_acc_en)              | False    |
 * | Template   | L_CACHED            | Read L through the math RISC's L1 data cache, enabled only for the duration of the call | bool | true (default), false                               | False    |
 * | Function   | cb_l                | Circular buffer holding the L tile                              | uint32_t        | 0 to 31                                                    | True     |
 * | Function   | l_tile_idx          | Index of the L tile within cb_l, relative to its front          | uint32_t        | Less than the number of front-waited tiles                 | True     |
 * | Function   | idst_in             | DST register index of the right-hand side                       | uint32_t        | Less than the size of the DST register buffer              | True     |
 * | Function   | idst_out            | DST register index that receives X                              | uint32_t        | Less than the size of the DST register buffer, not idst_in | True     |
 */
// clang-format on
template <
    DataFormat L_FORMAT = DataFormat::Float32,
    bool L_NEGATED = false,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    bool L_CACHED = true>
ALWI void triangle_solve_tile(
    uint32_t cb_l, uint32_t l_tile_idx, [[maybe_unused]] uint32_t idst_in, [[maybe_unused]] uint32_t idst_out) {
    static_assert(is_fp32_dest_acc_en, "triangle_solve_tile needs 32-bit destination accumulation (fp32_dest_acc_en)");
    // The MATH RISC reads a full 32x32 tile with L_FORMAT's element size straight from L1, so the CB's data
    // format must be L_FORMAT, its tiles 32x32, and the tile must lie within the CB.
    UNPACK({
        const uint32_t operand_id = get_operand_id(cb_l);
        LLK_ASSERT(
            get_operand_src_format(operand_id) == static_cast<uint32_t>(L_FORMAT),
            "triangle_solve_tile: cb_l's data format is not L_FORMAT");
        LLK_ASSERT(get_operand_num_faces(operand_id) == 4, "triangle_solve_tile: cb_l's tiles must have 4 faces");
        LLK_ASSERT(
            get_operand_face_r_dim(operand_id) == FACE_R_DIM, "triangle_solve_tile: cb_l's faces must be 16 rows");
        LLK_ASSERT(cb_access_within_bounds(operand_id, l_tile_idx, 1), "triangle_solve_tile: l_tile_idx exceeds cb_l");
    })
    // UNPACK resolves the tile's L1 address and mailboxes it to MATH and PACK.
    const uint32_t l1_base = get_tile_address(cb_l, l_tile_idx);
    MATH((llk_math_triangle_solve_sfpu_tile<L_FORMAT, L_NEGATED, L_CACHED>(l1_base, idst_in, idst_out)));
    // MATH reads L from L1 itself, which nothing orders against UNPACK's later cb_pop_front of cb_l: MATH signals
    // when its reads are done and UNPACK waits for that before returning.
    constexpr uint32_t l_reads_done = 1;  // mailbox token; only its arrival matters
    MATH((mailbox_write(ckernel::ThreadId::UnpackThreadId, l_reads_done)));
    UNPACK((mailbox_read(ckernel::ThreadId::MathThreadId)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void triangle_solve_tile_init() { MATH((llk_math_triangle_solve_sfpu_init())); }

}  // namespace ckernel
