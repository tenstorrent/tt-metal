// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Streamed matmul (TRISC) for the decode stream ops (fused_decode/stream.py): one 1x32 output tile per weight column,
// out[col] = in0 . W[:, col] over kt K tiles (1x32 activation tiles, custom_mm, LoFi, FP32 accumulation). The down
// stream uses it for the score-weighted expert sum (K = k segments [w_e * act_e | w_e]), LinearStream for a dense
// projection + bias.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"
#ifdef W_FP32_FOLD
#include "api/compute/eltwise_unary/binop_with_scalar.h"
// split_acc keeps two FP32 accumulators per output row (dest rows 0 / 8 of each face: alternate K faces);
// custom_mm's own finalize moves one through SrcB (TF32, ~10-bit mantissa) before adding it. custom_mm_block<false>
// leaves them apart and this folds them in FP32 on the SFPU: rows 8.. of each face onto rows 0.. (VectorMode::R).
#ifdef TRISC_MATH
#include "sfpi.h"
namespace ckernel::sfpu {
template <int ITERATIONS>
inline void calculate_split_fold() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat lo = sfpi::dst_reg[0];
        const sfpi::vFloat hi = sfpi::dst_reg[4];  // 8 dest rows further
        sfpi::dst_reg[0] = lo + hi;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif
ALWI void fold_rows(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_split_fold, (2), idst, VectorMode::R));
}
#endif

#if defined(EXP_TAIL_SUBS) && (defined(W_FP32_FOLD) || defined(W_TRANSPOSED))
#error "EXP_TAIL_SUBS: the plain (finalizing) BFP4 / BFP8 down stream only"
#endif

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    constexpr uint32_t cols = get_compile_time_arg_val(4);

#ifdef W_TRANSPOSED
    constexpr bool transpose = true;  // transposed BFP4 tiles (GPT_OSS_DECODE_SPLIT_NATIVE)
#else
    constexpr bool transpose = false;
#endif
#ifdef W_NO_SPLIT_ACC
    constexpr bool split_acc = false;
#else
    constexpr bool split_acc = true;
#endif
    constexpr bool dense_packing = false;
    custom_mm_block_init<transpose, split_acc, dense_packing>(cb_in0, cb_w, cb_out);

    cb_wait_front(cb_in0, kt);
    for (uint32_t c = 0; c < cols; ++c) {
        tile_regs_acquire();
#ifdef EXP_TAIL_SUBS
        // TEMP experiment (fused_decode/config.py DECODE_EXPERT_TAIL_SPLIT, the expert down stream): the last column,
        // pushed by the reader (stream_down_reader.cpp) in EXP_TAIL_SUBS sub-blocks, is multiplied sub-block by
        // sub-block as they land; every K tile accumulates into the same DST rows whatever the call split, and only
        // the last call finalizes (the same MVMUL / ELWADD sequence as one call).
        if (c + 1 == cols) {
            constexpr uint32_t tt = EXP_TAIL_TILES;
            for (uint32_t t = 0; t + 1 < EXP_TAIL_SUBS; ++t) {
                cb_wait_front(cb_w, (t + 1) * tt);
                custom_mm_block<false>(cb_in0, cb_w, t * tt, t * tt, 0, tt);
            }
            cb_wait_front(cb_w, kt);
            custom_mm_block<true>(cb_in0, cb_w, kt - tt, kt - tt, 0, tt);
            cb_pop_front(cb_w, kt);
        } else
#endif
        {
            cb_wait_front(cb_w, kt);
#ifdef W_FP32_FOLD
            custom_mm_block<false>(cb_in0, cb_w, 0, 0, 0, kt);
            cb_pop_front(cb_w, kt);
            binop_with_scalar_tile_init();
            fold_rows(0);
#else
            custom_mm_block<true>(cb_in0, cb_w, 0, 0, 0, kt);
            cb_pop_front(cb_w, kt);
#endif
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
#ifdef W_FP32_FOLD
        custom_mm_block_init_short<transpose, split_acc, dense_packing>(cb_in0, cb_w, cb_out);  // after the SFPU
#endif
    }
    cb_pop_front(cb_in0, kt);
    custom_mm_block_uninit<dense_packing>();
}
