// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
#include "llk_assert.h"
#include "sanitizer/api.h"
#ifdef TRISC_MATH
#include "llk_math_matmul_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_matmul_api.h"
#include "llk_unpack_common_api.h"
#endif
// defines the default throttle level for matmul kernels (default 0)
#ifndef MM_THROTTLE
#define MM_THROTTLE 0
#endif
namespace ckernel {

#ifdef ARCH_BLACKHOLE
// defines the FW-controlled throttle level for block matmul kernels on Blackhole
#define MM_THROTTLE_MAX 5
// 4-byte word at MEM_L1_ARC_FW_SCRATCH written by FW - even means no throttle, odd means throttle
volatile tt_l1_ptr std::uint32_t* throttle_ptr =
    reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(MEM_L1_ARC_FW_SCRATCH);
// tracks the state of the currently programmed matmul MOP (0: default throttle level, 1: max throttle level)
static std::uint32_t throttled_mop_status = 0;

// clang-format off
/**
 * Performs matmul block operation with dynamic throttling.
 * This function is only available on Blackhole architecture and implements
 * firmware-controlled dynamic throttling for block matmul operations.
 * The throttle level is controlled by firmware via MEM_L1_ARC_FW_SCRATCH.
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)                  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)                 | uint32_t | 0 to 31                                        | True     |
 * | idst           | The index of the tile in DST REG to which the result C will be written. | uint32_t | Must be less than the acquired size of DST REG | True     |
 * | transpose      | The transpose flag for performing transpose operation on tiles in B.    | bool     | Must be true or false                          | True     |
 * | ct_dim         | The column dimension for the output block.                              | uint32_t | Must be equal to block B column dimension      | True     |
 * | rt_dim         | The row dimension for the output block.                                 | uint32_t | Must be equal to block A row dimension         | True     |
 * | kt_dim         | The inner dimension.                                                    | uint32_t | Must be equal to block A column dimension      | True     |
 */
// clang-format on
template <bool row_mop = false>
ALWI void matmul_block_math_dynamic_throttle(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    std::uint32_t idst,
    const std::uint32_t transpose,
    std::uint32_t ct_dim,
    std::uint32_t rt_dim) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    // Dynamic throttling is only available on Blackhole architecture
    // Check firmware-controlled throttle enable flag (even = no throttle, odd = throttle)
    volatile std::uint32_t mm_throttle_en = *(throttle_ptr) % 2;
    if (mm_throttle_en) {
        if (throttled_mop_status != 1) {
            MATH((
                llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE_MAX>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
            throttled_mop_status = 1;
        }
        MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE_MAX>(idst, ct_dim, rt_dim)));
    } else {
        if (throttled_mop_status != 0) {
            MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE, row_mop>(
                in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
            throttled_mop_status = 0;
        }
        MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE, 4, row_mop>(idst, ct_dim, rt_dim)));
    }
#endif
}
#endif

// clang-format off
/**
 * Short init for matmul_tiles. Configures the unpacker and math engine to matmul mode.
 *
 * Must be called before matmul_tiles. The one-time HW configuration must already have been
 * performed via compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out) at the start of MAIN.
 * Matmul maps in0 -> SrcB and in1 -> SrcA (the reverse of other ops), which is why
 * compute_kernel_hw_startup must use SrcOrder::Reverse.
 *
 * NOTE (known gap, #46769): if a preceding op left SrcA/SrcB with asymmetric tile sizes (i.e. different
 * data formats per source) and the following matmul uses the same formats, matmul_init cannot fix the
 * per-source tile sizes on its own. It does not re-program the tile descriptor, and a reconfig_data_format
 * is inappropriate when the data formats did not change. No current kernel hits this; tracked in #46769.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range                                       | Required |
 * |----------------|---------------------------------------------------------------|----------|---------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)        | uint32_t | 0 to 31                                           | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)       | uint32_t | 0 to 31                                           | True     |
 * | transpose      | The transpose flag for performing transpose operation on B    | uint32_t | Any positive value will indicate transpose is set | False    |
 */
// clang-format on
ALWI void matmul_init(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    const std::uint32_t transpose = 0,
    std::uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    state_configure(in1_cb_id, in0_cb_id, call_line);
    MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose)));
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose)));
#else
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose)));
    MATH((llk_math_matmul_init<MATH_FIDELITY>(in0_cb_id, in1_cb_id)));
#endif
}

// clang-format off
/**
 * (Quasar) Undo the automatic MxFp4 -> MxFp4_2x_B src-format selection applied by matmul_init.
 *
 * matmul_init overrides an MxFp4 operand's unpacker OUT_DATA_FORMAT and ALU format to the 2x-packed
 * MxFp4_2x_B, diverging from the op-agnostic unpack_dst_format[] table. That override PERSISTS: the
 * non-matmul unpack inits never reprogram OUT_DATA_FORMAT, and reconfig_data_format is silently
 * skipped for a same-format operand. So a kernel that feeds the SAME MxFp4 buffer to matmul and then
 * to a non-matmul op (datacopy/SFPU/eltwise) MUST call mm_uninit(in0, in1) after the matmuls and
 * before the next op, or that op will keep unpacking the buffer as MxFp4_2x_B and produce garbage.
 * A no-op on non-MxFp4 operands and on non-Quasar architectures.
 *
 * | Argument  | Description                                            | Type     | Valid Range | Required |
 * |-----------|--------------------------------------------------------|----------|-------------|----------|
 * | in0_cb_id | First input CB used in the matmul (same as matmul_init)| uint32_t | 0 to 31     | True     |
 * | in1_cb_id | Second input CB used in the matmul                     | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void mm_uninit(std::uint32_t in0_cb_id, std::uint32_t in1_cb_id) {
#ifdef ARCH_QUASAR
    UNPACK((llk_unpack_AB_matmul_uninit(in0_cb_id, in1_cb_id)));
    MATH((llk_math_matmul_uninit(in0_cb_id, in1_cb_id)));
#endif
}

// clang-format off
/**
 * Performs tile-sized matrix multiplication *C=A\*B* between the tiles in two
 * specified input CBs and accumulates the result to DST (DST += C). The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking and
 * is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)                  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)                 | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of the tile A from the first input CB                         | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of the tile B from the second input CB                        | uint32_t | Must be less than the size of the CB           | True     |
 * | idst           | The index of the tile in DST REG to which the result C will be written. | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
ALWI void matmul_tiles(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    std::uint32_t in0_tile_index,
    std::uint32_t in1_tile_index,
    std::uint32_t idst) {
    LLK_SAN_FUNCTION();
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index)));
#ifndef ARCH_QUASAR
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst)));
#else
    MATH((llk_math_matmul_tile(idst)));
#endif
}

// clang-format off
/**
 * Short init for matmul_block. Configures the unpacker and math engine to matmul mode.
 *
 * Must be called before matmul_block. The one-time HW configuration must already have been
 * performed via compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out) at the start of MAIN.
 * Matmul maps in0 -> SrcB and in1 -> SrcA (the reverse of other ops), which is why
 * compute_kernel_hw_startup must use SrcOrder::Reverse.
 *
 * NOTE (known gap, #46769): if a preceding op left SrcA/SrcB with asymmetric tile sizes (i.e. different
 * data formats per source) and the following matmul uses the same formats, matmul_block_init cannot fix
 * the per-source tile sizes on its own. It does not re-program the tile descriptor, and a
 * reconfig_data_format is inappropriate when the data formats did not change. No current kernel hits this;
 * tracked in #46769.
 *
 * The template parameter row_mop (Blackhole; other architectures ignore it) makes the math thread run one MOP per
 * reuse row of full 32x32 tiles when MM_THROTTLE is 0; matmul_block must be called with the same value. That MOP also
 * uses the math thread's ADDR_MOD_3, 6 and 7, so a math-thread SFPU init between the two needs a new matmul_block_init,
 * and a math-thread SFPU op after the matmul needs its own init again.
 *
 * Return value: None
 *
 * | Argument  | Description                                                | Type     | Valid Range                                                                                    | Required |
 * |-----------|------------------------------------------------------------|----------|------------------------------------------------------------------------------------------------|----------|
 * | in0_cb_id | The identifier of the first input circular buffer (CB)     | uint32_t | 0 to 31                                                                                        | True     |
 * | in1_cb_id | The identifier of the second input circular buffer (CB)    | uint32_t | 0 to 31                                                                                        | True     |
 * | transpose | The transpose flag for performing transpose operation on B | uint32_t | Any positive value will indicate transpose is set                                              | False    |
 * | ct_dim    | The column dimension for the output block.                 | uint32_t | Must be equal to block B column dimension; 1 to 8 in half-sync mode, 1 to 16 in full-sync mode | False    |
 * | rt_dim    | The row dimension for the output block.                    | uint32_t | Must be equal to block A row dimension; 1 to 8 in half-sync mode, 1 to 16 in full-sync mode    | False    |
 * | kt_dim    | The inner dimension.                                       | uint32_t | Must be equal to block A column dimension                                                      | False    |
 */
// clang-format on
template <bool row_mop = false>
ALWI void matmul_block_init(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    const std::uint32_t transpose = 0,
    std::uint32_t ct_dim = 1,
    std::uint32_t rt_dim = 1,
    std::uint32_t kt_dim = 1,
    std::uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    state_configure(in1_cb_id, in0_cb_id, call_line);
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)));
#ifdef ARCH_BLACKHOLE
    MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE, row_mop>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
    // Dynamic throttling is only available on Blackhole architecture
    MATH((throttled_mop_status = 0));
#else
    MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
#endif
#else
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_init<MATH_FIDELITY>(in0_cb_id, in1_cb_id, ct_dim, rt_dim)));
#endif
}

// clang-format off
/**
 * Performs block-sized matrix multiplication *C=A\*B* between the blocks in two
 * different input CBs and accumulates the result to DST (DST += C). The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking and
 * is only available on the compute engine.
 *
 * A block is a rectangle of tiles: A is rt_dim x kt_dim tiles, B is kt_dim x ct_dim tiles, and the
 * output C is rt_dim x ct_dim tiles. So a block is just ct_dim * rt_dim output tiles produced in one
 * call (with kt_dim tiles along the shared inner dimension). The output must fit in DST, so the block
 * size is limited by DST size and sync mode (see matmul_block_init for the valid ct_dim/rt_dim ranges).
 * A call may use smaller ct_dim and rt_dim than matmul_block_init, but ct_dim >= rt_dim must hold for it exactly
 * when it held for the init (the init fixes which operand is held); a block of the other direction needs a new init.
 * The template parameter row_mop must be the value matmul_block_init was called with.
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)                  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)                 | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of the tile in block A from the first input CB                | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of the tile in block B from the second input CB               | uint32_t | Must be less than the size of the CB           | True     |
 * | idst           | The index of the tile in DST REG to which the result C will be written. | uint32_t | Must be less than the acquired size of DST REG | True     |
 * | transpose      | The transpose flag for performing transpose operation on tiles in B.    | bool     | Must be true or false                          | True     |
 * | ct_dim         | The column dimension for the output block.                              | uint32_t | Must be equal to block B column dimension      | True     |
 * | rt_dim         | The row dimension for the output block.                                 | uint32_t | Must be equal to block A row dimension         | True     |
 * | kt_dim         | The inner dimension.                                                    | uint32_t | Must be equal to block A column dimension      | True     |
 */
// clang-format on
template <bool row_mop = false>
ALWI void matmul_block(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    std::uint32_t in0_tile_index,
    std::uint32_t in1_tile_index,
    std::uint32_t idst,
    const std::uint32_t transpose,
    std::uint32_t ct_dim,
    std::uint32_t rt_dim,
    std::uint32_t kt_dim,
    std::uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    state_configure(in1_cb_id, in0_cb_id, call_line);
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim)));
#ifdef ARCH_BLACKHOLE
    // Dynamic throttling is only available on Blackhole architecture
    MATH((matmul_block_math_dynamic_throttle<row_mop>(in0_cb_id, in1_cb_id, idst, transpose, ct_dim, rt_dim)));
#else
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst, ct_dim, rt_dim)));
#endif
#else
    LLK_ASSERT(idst == 0, "non-default idst not supported on Quasar");
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_block(ct_dim, rt_dim)));
#endif
}

// clang-format off
/**
 * Performs matmul_block for k_steps consecutive steps along the inner dimension, the same as
 *   for (k = 0; k < k_steps; k++)
 *       matmul_block<row_mop>(in0_cb_id, in1_cb_id, in0_tile_index + k * in0_k_stride, in1_tile_index + k * in1_k_stride,
 *                             idst, transpose, ct_dim, rt_dim, kt_dim);
 * Every tile of the k steps must be in the input CBs. On Blackhole a one-tile block (ct_dim and rt_dim 1) takes one
 * unpacker config context for all the k steps instead of one per k step.
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)                  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)                 | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of the first k step's tile in block A                         | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of the first k step's tile in block B                         | uint32_t | Must be less than the size of the CB           | True     |
 * | idst           | The index of the tile in DST REG to which the result C will be written. | uint32_t | Must be less than the acquired size of DST REG | True     |
 * | transpose      | The transpose flag for performing transpose operation on tiles in B.    | bool     | Must be true or false                          | True     |
 * | ct_dim         | The column dimension for the output block.                              | uint32_t | As for matmul_block                            | True     |
 * | rt_dim         | The row dimension for the output block.                                 | uint32_t | As for matmul_block                            | True     |
 * | kt_dim         | The inner dimension (the in0 tile stride from one row to the next).     | uint32_t | As for matmul_block                            | True     |
 * | k_steps        | The number of k steps.                                                  | uint32_t | At least 1                                     | True     |
 * | in0_k_stride   | The in0 tiles from one k step to the next.                              | uint32_t | Any                                            | True     |
 * | in1_k_stride   | The in1 tiles from one k step to the next.                              | uint32_t | Any                                            | True     |
 */
// clang-format on
template <bool row_mop = false>
ALWI void matmul_block_k_loop(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    std::uint32_t in0_tile_index,
    std::uint32_t in1_tile_index,
    std::uint32_t idst,
    const std::uint32_t transpose,
    std::uint32_t ct_dim,
    std::uint32_t rt_dim,
    std::uint32_t kt_dim,
    std::uint32_t k_steps,
    std::uint32_t in0_k_stride,
    std::uint32_t in1_k_stride,
    std::uint32_t call_line = __builtin_LINE()) {
#ifdef ARCH_BLACKHOLE
    LLK_SAN_FUNCTION();
    state_configure(in1_cb_id, in0_cb_id, call_line);
    UNPACK((llk_unpack_AB_matmul_k_loop(
        in0_cb_id,
        in1_cb_id,
        in0_tile_index,
        in1_tile_index,
        ct_dim,
        rt_dim,
        kt_dim,
        k_steps,
        in0_k_stride,
        in1_k_stride)));
    for (std::uint32_t k = 0; k < k_steps; k++) {
        MATH((matmul_block_math_dynamic_throttle<row_mop>(in0_cb_id, in1_cb_id, idst, transpose, ct_dim, rt_dim)));
    }
#else
    for (std::uint32_t k = 0; k < k_steps; k++) {
        matmul_block<row_mop>(
            in0_cb_id,
            in1_cb_id,
            in0_tile_index + k * in0_k_stride,
            in1_tile_index + k * in1_k_stride,
            idst,
            transpose,
            ct_dim,
            rt_dim,
            kt_dim,
            call_line);
    }
#endif
}

}  // namespace ckernel
