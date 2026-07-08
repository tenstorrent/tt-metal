// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
volatile tt_l1_ptr uint32_t* throttle_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MEM_L1_ARC_FW_SCRATCH);
// tracks the state of the currently programmed matmul MOP (0: default throttle level, 1: max throttle level)
static uint32_t throttled_mop_status = 0;

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
ALWI void matmul_block_math_dynamic_throttle(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t idst, const uint32_t transpose, uint32_t ct_dim, uint32_t rt_dim) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    // Dynamic throttling is only available on Blackhole architecture
    // Check firmware-controlled throttle enable flag (even = no throttle, odd = throttle)
    volatile uint32_t mm_throttle_en = *(throttle_ptr) % 2;
    if (mm_throttle_en) {
        if (throttled_mop_status != 1) {
            MATH((
                llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE_MAX>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
            throttled_mop_status = 1;
        }
        MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE_MAX>(idst, ct_dim, rt_dim)));
    } else {
        if (throttled_mop_status != 0) {
            MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
            throttled_mop_status = 0;
        }
        MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst, ct_dim, rt_dim)));
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
    uint32_t in0_cb_id, uint32_t in1_cb_id, const uint32_t transpose = 0, uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    state_configure(in1_cb_id, in0_cb_id, call_line);
    MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose)));
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose)));
#else
    LLK_ASSERT(transpose == 0, "non-default transpose not supported on Quasar");
    UNPACK((llk_unpack_AB_matmul_init<false /*transpose*/>(in0_cb_id, in1_cb_id)));
    MATH((llk_math_matmul_init<MATH_FIDELITY>(in0_cb_id, in1_cb_id)));
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
 * | dst_tile_index | The index of the tile in DST REG to which the result C will be written. | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
ALWI void matmul_tiles(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in0_tile_index, uint32_t in1_tile_index, uint32_t idst) {
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
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range                                         | Required |
 * |----------------|---------------------------------------------------------------|----------|-----------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)        | uint32_t | 0 to 31                                             | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)       | uint32_t | 0 to 31                                             | True     |
 * | transpose      | The transpose flag for performing transpose operation on B    | uint32_t | Any positive value will indicate transpose is set   | False    |
 * | ct_dim         | The column dimension for the output block.                    | uint32_t | Must be equal to block B column dimension           | False    |
 * | rt_dim         | The row dimension for the output block.                       | uint32_t | Must be equal to block A row dimension              | False    |
 * | kt_dim         | The inner dimension.                                          | uint32_t | Must be equal to block A column dimension           | False    |
 */
// clang-format on
ALWI void matmul_block_init(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    const uint32_t transpose = 0,
    uint32_t ct_dim = 1,
    uint32_t rt_dim = 1,
    uint32_t kt_dim = 1,
    uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    state_configure(in1_cb_id, in0_cb_id, call_line);
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
#ifdef ARCH_BLACKHOLE
    // Dynamic throttling is only available on Blackhole architecture
    MATH((throttled_mop_status = 0));
#endif
#else
    LLK_ASSERT(transpose == 0, "non-default transpose not supported on Quasar");
    UNPACK((llk_unpack_AB_matmul_init<false /*transpose*/>(in0_cb_id, in1_cb_id, ct_dim, rt_dim, kt_dim)));
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
 * size is limited by DST size and sync mode (see mm_block_init for the valid ct_dim/rt_dim ranges).
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
ALWI void matmul_block(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t in0_tile_index,
    uint32_t in1_tile_index,
    uint32_t idst,
    const uint32_t transpose,
    uint32_t ct_dim,
    uint32_t rt_dim,
    uint32_t kt_dim,
    uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    state_configure(in1_cb_id, in0_cb_id, call_line);
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim)));
#ifdef ARCH_BLACKHOLE
    // Dynamic throttling is only available on Blackhole architecture
    MATH((matmul_block_math_dynamic_throttle(in0_cb_id, in1_cb_id, idst, transpose, ct_dim, rt_dim)));
#else
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst, ct_dim, rt_dim)));
#endif
#else
    LLK_ASSERT(transpose == 0, "non-default transpose not supported on Quasar");
    LLK_ASSERT(idst == 0, "non-default idst not supported on Quasar");
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_block(ct_dim, rt_dim)));
#endif
}

// =====================================================================================================================
// Deprecated API
//
// The functions below implement the old matmul programming model. The new model is:
//   compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out);  // once at the start of MAIN
//   matmul_init(in0, in1, transpose);                             // before matmul_tiles
//   matmul_block_init(in0, in1, transpose, ct_dim, rt_dim, kt_dim); // before matmul_block
// Generic data-format reconfiguration is done via reconfig_data_format_srca / reconfig_data_format
// (from reconfig_data_format.h). Note in1 feeds SrcA and in0 feeds SrcB.
// =====================================================================================================================

// clang-format off
/**
 * Initialization for matmul_tiles operation. Must be called before matmul_tiles.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range                                        | Required |
 * |----------------|---------------------------------------------------------------|----------|----------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)        | uint32_t | 0 to 31                                            | False    |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)       | uint32_t | 0 to 31                                            | False    |
 * | out_cb_id      | The identifier of the output circular buffer (CB)             | uint32_t | 0 to 31                                            | False    |
 * | transpose      | The transpose flag for performing transpose operation on B    | uint32_t | Any positive value will indicate transpose is set  | False    |
 */
// clang-format on
[[deprecated(
    "Use compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out) once at kernel start, then matmul_init(in0, in1, "
    "transpose).")]] ALWI void
mm_init(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t out_cb_id,
    const uint32_t transpose = 0,
    uint32_t call_line = __builtin_LINE()) {
#ifndef ARCH_QUASAR
    state_configure(in1_cb_id, in0_cb_id, out_cb_id, call_line);
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(in1_cb_id, in0_cb_id)));
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose)));

    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(in0_cb_id, in1_cb_id)));
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose)));

    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(out_cb_id)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, PackMode::Default>()));
    PACK((llk_pack_init(out_cb_id)));
#else
    LLK_ASSERT(transpose == 0, "non-default transpose not supported on Quasar");
    UNPACK((llk_unpack_hw_configure(in1_cb_id, in0_cb_id)));
    UNPACK((llk_unpack_AB_matmul_init<false /*transpose*/>(in0_cb_id, in1_cb_id)));

    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(in0_cb_id, in1_cb_id)));
    MATH((llk_math_pack_sync_init()));
    MATH((llk_math_matmul_init<MATH_FIDELITY>(in0_cb_id, in1_cb_id)));

    PACK((llk_pack_hw_configure(out_cb_id)));
    PACK((llk_pack_init(out_cb_id)));
    PACK((llk_pack_dest_init()));
#endif
}

// clang-format off
/**
 * Performs tile-sized matrix multiplication *C=A\*B* between the tiles
 * located in SRCA and SRCB and accumulates the result to DST (DST += C). The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking and
 * is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | idst           | The index of the tile in DST REG to which the result C will be written. | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
#ifndef ARCH_QUASAR
template <uint32_t num_faces = 4>
[[deprecated("Unused; slated for removal. Use matmul_tiles() instead.")]] ALWI void matmul_tiles_math(uint32_t idst) {
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE, num_faces>(idst)));
}
#endif

// clang-format off
/**
 * A short version of matmul_tiles initialization.
 * Configure the unpacker and math engine to matmul mode.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range                                       | Required |
 * |----------------|---------------------------------------------------------------|----------|---------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)        | uint32_t | 0 to 31                                           | False    |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)       | uint32_t | 0 to 31                                           | False    |
 * | transpose      | The transpose flag for performing transpose operation on B    | uint32_t | Any positive value will indicate transpose is set | False    |
 */
// clang-format on
[[deprecated("Renamed to matmul_init().")]] ALWI void mm_init_short(
    uint32_t in0_cb_id, uint32_t in1_cb_id, const uint32_t transpose = 0, uint32_t call_line = __builtin_LINE()) {
    matmul_init(in0_cb_id, in1_cb_id, transpose, call_line);
}

// clang-format off
/**
 * A short version of matmul_tiles initialization.
 * It is used to reconfigure srcA of the compute engine back to matmul mode.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range                                       | Required |
 * |----------------|---------------------------------------------------------------|----------|---------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)        | uint32_t | 0 to 31                                           | False    |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)       | uint32_t | 0 to 31                                           | False    |
 * | c_in_old_srca  | The identifier of the old input to src A circular buffer (CB) | uint32_t | 0 to 31                                           | False    |
 * | transpose      | The transpose flag for performing transpose operation on B    | uint32_t | Any positive value will indicate transpose is set | False    |
 */
// clang-format on
#ifndef ARCH_QUASAR
[[deprecated("Call reconfig_data_format_srca(old_srca, in1) then matmul_init(in0, in1, transpose).")]] ALWI void
mm_init_short_with_dt(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t c_in_old_srca, const uint32_t transpose = 0) {
    reconfig_data_format_srca(c_in_old_srca, in1_cb_id);
    matmul_init(in0_cb_id, in1_cb_id, transpose);
}
#endif

// clang-format off
/**
 * Initialization for matmul_block operation. Must be called before matmul_block.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range                                         | Required |
 * |----------------|---------------------------------------------------------------|----------|-----------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)        | uint32_t | 0 to 31                                             | False    |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)       | uint32_t | 0 to 31                                             | False    |
 * | out_cb_id      | The identifier of the output circular buffer (CB)             | uint32_t | 0 to 31                                             | False    |
 * | ct_dim         | The number of columns of the output matrix in tiles           | uint32_t | 1 to 8 in half-sync mode, 1 to 16 in full-sync mode | False    |
 * | rt_dim         | The number of rows of the output matrix in tiles              | uint32_t | 1 to 8 in half-sync mode, 1 to 16 in full-sync mode | False    |
 * | kt_dim         | The inner dim of the input matrices in tiles                  | uint32_t | 1 to 2^32-1                                         | False    |
 */
// clang-format on
[[deprecated(
    "Use compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out) once at kernel start, then matmul_block_init(in0, "
    "in1, transpose, ct_dim, rt_dim, kt_dim).")]] ALWI void
mm_block_init(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t out_cb_id,
    const uint32_t transpose = 0,
    uint32_t ct_dim = 1,
    uint32_t rt_dim = 1,
    uint32_t kt_dim = 1,
    uint32_t call_line = __builtin_LINE()) {
#ifndef ARCH_QUASAR
    state_configure(in1_cb_id, in0_cb_id, out_cb_id, call_line);

    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(in1_cb_id, in0_cb_id)));
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)));

    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(in0_cb_id, in1_cb_id)));
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
#ifdef ARCH_BLACKHOLE
    // Dynamic throttling is only available on Blackhole architecture
    MATH((throttled_mop_status = 0));
#endif

    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(out_cb_id)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, PackMode::Default>()));
    PACK((llk_pack_init<PackMode::Default, false /* zero_output */>(out_cb_id)));
#else
    LLK_ASSERT(transpose == 0, "non-default transpose not supported on Quasar");
    UNPACK((llk_unpack_hw_configure(in1_cb_id, in0_cb_id)));
    UNPACK((llk_unpack_AB_matmul_init<false /*transpose*/>(in0_cb_id, in1_cb_id, ct_dim, rt_dim, kt_dim)));

    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(in0_cb_id, in1_cb_id)));
    MATH((llk_math_pack_sync_init()));
    MATH((llk_math_matmul_init<MATH_FIDELITY>(in0_cb_id, in1_cb_id, ct_dim, rt_dim)));

    PACK((llk_pack_hw_configure(out_cb_id)));
    PACK((llk_pack_init(out_cb_id)));
    PACK((llk_pack_dest_init()));
#endif
}

// clang-format off
/**
 * A short version of matmul_block initialization.
 * Configure the unpacker and math engine to matmul mode.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range                                         | Required |
 * |----------------|---------------------------------------------------------------|----------|-----------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)        | uint32_t | 0 to 31                                             | False    |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)       | uint32_t | 0 to 31                                             | False    |
 * | transpose      | The transpose flag for performing transpose operation on B    | uint32_t | Any positive value will indicate transpose is set   | False    |
 * | ct_dim         | The column dimension for the output block.                    | uint32_t | Must be equal to block B column dimension           | False    |
 * | rt_dim         | The row dimension for the output block.                       | uint32_t | Must be equal to block A row dimension              | False    |
 * | kt_dim         | The inner dimension.                                          | uint32_t | Must be equal to block A column dimension           | False    |
 */
// clang-format on
[[deprecated("Renamed to matmul_block_init().")]] ALWI void mm_block_init_short(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    const uint32_t transpose = 0,
    uint32_t ct_dim = 1,
    uint32_t rt_dim = 1,
    uint32_t kt_dim = 1,
    uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
    matmul_block_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim, call_line);
}

// clang-format off
/**
 * A short version of matmul_block initialization.
 * It is used to reconfigure srcA of the compute engine back to matmul mode.
 *
 * Return value: None
 *
 * | Argument       | Description                                                | Type     | Valid Range                               | Required |
 * |----------------|------------------------------------------------------------|----------|-------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)     | uint32_t | 0 to 31                                   | False    |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)    | uint32_t | 0 to 31                                   | False    |
 * | old_in1_cb_id  | The identifier of the old in1_cb_id circular buffer (CB)   | uint32_t | 0 to 31                                   | False    |
 * | ct_dim         | The column dimension for the output block.                 | uint32_t | Must be equal to block B column dimension | False    |
 * | rt_dim         | The row dimension for the output block.                    | uint32_t | Must be equal to block A row dimension    | False    |
 * | kt_dim         | The inner dimension.                                       | uint32_t | Must be equal to block A column dimension | False    |
 */
// clang-format on
#ifndef ARCH_QUASAR
[[deprecated("Call reconfig_data_format_srca(old_in1, in1) then matmul_block_init(...).")]] ALWI void
mm_block_init_short_with_dt(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t old_in1_cb_id,
    const uint32_t transpose = 0,
    uint32_t ct_dim = 1,
    uint32_t rt_dim = 1,
    uint32_t kt_dim = 1,
    uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
    reconfig_data_format_srca(old_in1_cb_id, in1_cb_id);
    matmul_block_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim, call_line);
}
#endif

// clang-format off
/**
 * A short version of matmul_block initialization.
 * It is used to reconfigure srcA and srcB of the compute engine back to matmul mode.
 *
 * Return value: None
 *
 * | Argument       | Description                                                | Type     | Valid Range                               | Required |
 * |----------------|------------------------------------------------------------|----------|-------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)     | uint32_t | 0 to 31                                   | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)    | uint32_t | 0 to 31                                   | True     |
 * | old_in0_cb_id  | The identifier of the old in0_cb_id circular buffer (CB)   | uint32_t | 0 to 31                                   | True     |
 * | old_in1_cb_id  | The identifier of the old in1_cb_id circular buffer (CB)   | uint32_t | 0 to 31                                   | True     |
 * | ct_dim         | The column dimension for the output block.                 | uint32_t | Must be equal to block B column dimension | False    |
 * | rt_dim         | The row dimension for the output block.                    | uint32_t | Must be equal to block A row dimension    | False    |
 * | kt_dim         | The inner dimension.                                       | uint32_t | Must be equal to block A column dimension | False    |
 */
// clang-format on
#ifndef ARCH_QUASAR
[[deprecated("Call reconfig_data_format(old_in1, in1, old_in0, in0) then matmul_block_init(...).")]] ALWI void
mm_block_init_short_with_both_dt(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t old_in0_cb_id,
    uint32_t old_in1_cb_id,
    const uint32_t transpose = 0,
    uint32_t ct_dim = 1,
    uint32_t rt_dim = 1,
    uint32_t kt_dim = 1) {
    LLK_SAN_FUNCTION();
    reconfig_data_format(old_in1_cb_id, in1_cb_id, old_in0_cb_id, in0_cb_id);
    matmul_block_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim);
}
#endif

}  // namespace ckernel
