// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
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
 * Internal helper for matmul_block: re-programs the matmul MOP at runtime based on the
 * firmware-controlled throttle flag stored at MEM_L1_ARC_FW_SCRATCH (even = no throttle,
 * odd = throttle). Only available on Blackhole. Called from MATH context only.
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
 */
// clang-format on
ALWI void matmul_block_math_dynamic_throttle(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t idst, const uint32_t transpose, uint32_t ct_dim, uint32_t rt_dim) {
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
#endif  // TODO: AM; add Quasar implementation
}
#endif

// clang-format off
/**
 * Per-op initialization for matmul. Configures the unpacker and math engine for matmul mode
 * for the provided input circular buffers and (optional) output block dimensions, and must
 * be called before any subsequent matmul_tiles or matmul_block call that uses the same
 * configuration. This is a lightweight init that does NOT perform hardware configuration
 * (MMIO writes such as hw_configure / pack_dest_init); the one-time hardware init must be
 * done up front via compute_kernel_hw_startup().
 *
 * The same API serves both tile-level and block-level matmul. For tile-level matmul (i.e.
 * matmul_tiles), call this with the default block dimensions (ct_dim = rt_dim = kt_dim = 1).
 * For block-level matmul (i.e. matmul_block), pass the output block geometry explicitly.
 *
 * Programming model recap (LLK contract): hw_start_init -> op_init -> execute. This function
 * is the op_init for matmul. If a different op runs in between two matmuls and reconfigures
 * the unpacker/math engine, mm_init must be called again before the next matmul. If the
 * data formats of the operands change between calls, use mm_init_with_dt or
 * mm_init_with_both_dt to fold in the data-format reconfiguration.
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                         | Required |
 * |----------------|-------------------------------------------------------------------------|----------|-----------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB) — operand A     | uint32_t | 0 to 31                                             | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB) — operand B    | uint32_t | 0 to 31                                             | True     |
 * | transpose      | The transpose flag for performing transpose operation on tiles in B    | uint32_t | Any positive value will indicate transpose is set   | False    |
 * | ct_dim         | The column dimension for the output block (in tiles)                   | uint32_t | 1 to 8 in half-sync mode, 1 to 16 in full-sync mode | False    |
 * | rt_dim         | The row dimension for the output block (in tiles)                      | uint32_t | 1 to 8 in half-sync mode, 1 to 16 in full-sync mode | False    |
 * | kt_dim         | The inner dimension of the input matrices (in tiles)                   | uint32_t | 1 to 2^32 - 1                                       | False    |
 */
// clang-format on
ALWI void mm_init(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    const uint32_t transpose = 0,
    uint32_t ct_dim = 1,
    uint32_t rt_dim = 1,
    uint32_t kt_dim = 1,
    uint32_t call_line = __builtin_LINE()) {
#ifndef ARCH_QUASAR
    state_configure(in1_cb_id, in0_cb_id, call_line);
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
#ifdef ARCH_BLACKHOLE
    // Dynamic throttling is only available on Blackhole architecture
    MATH((throttled_mop_status = 0));
#endif
#else
    ASSERT(transpose == 0);  // matmul transpose not yet implemented for Quasar
    UNPACK((llk_unpack_AB_matmul_init<false /*transpose*/>(in0_cb_id, in1_cb_id, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_init<MATH_FIDELITY>(ct_dim, rt_dim)));
#endif
}

// clang-format off
/**
 * Per-op initialization for matmul that also reconfigures the data format on srcA. Equivalent
 * to issuing reconfig_data_format_srca(c_in_old_srca, in1_cb_id) followed by mm_init(...).
 * Use this when the operand A circular buffer or its data format differs from what was last
 * configured, but operand B is unchanged. For data-format changes on both operands, use
 * mm_init_with_both_dt().
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                         | Required |
 * |----------------|-------------------------------------------------------------------------|----------|-----------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB) — operand A     | uint32_t | 0 to 31                                             | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB) — operand B    | uint32_t | 0 to 31                                             | True     |
 * | c_in_old_srca  | The identifier of the previously-configured srcA circular buffer (CB)  | uint32_t | 0 to 31                                             | True     |
 * | transpose      | The transpose flag for performing transpose operation on tiles in B    | uint32_t | Any positive value will indicate transpose is set   | False    |
 * | ct_dim         | The column dimension for the output block (in tiles)                   | uint32_t | 1 to 8 in half-sync mode, 1 to 16 in full-sync mode | False    |
 * | rt_dim         | The row dimension for the output block (in tiles)                      | uint32_t | 1 to 8 in half-sync mode, 1 to 16 in full-sync mode | False    |
 * | kt_dim         | The inner dimension of the input matrices (in tiles)                   | uint32_t | 1 to 2^32 - 1                                       | False    |
 */
// clang-format on
ALWI void mm_init_with_dt(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t c_in_old_srca,
    const uint32_t transpose = 0,
    uint32_t ct_dim = 1,
    uint32_t rt_dim = 1,
    uint32_t kt_dim = 1) {
#ifndef ARCH_QUASAR
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE>(c_in_old_srca, in1_cb_id)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(c_in_old_srca, in1_cb_id)));
    mm_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim);
#endif  // TODO: AM; add Quasar implementation
}

// clang-format off
/**
 * Per-op initialization for matmul that also reconfigures the data format on both srcA and srcB.
 * Equivalent to issuing reconfig_data_format(old_in1_cb_id, in1_cb_id, old_in0_cb_id, in0_cb_id)
 * followed by mm_init(...). Use this when both operand circular buffers (or their data formats)
 * differ from what was last configured.
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                         | Required |
 * |----------------|-------------------------------------------------------------------------|----------|-----------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB) — operand A     | uint32_t | 0 to 31                                             | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB) — operand B    | uint32_t | 0 to 31                                             | True     |
 * | old_in0_cb_id  | The identifier of the previously-configured srcA circular buffer (CB)  | uint32_t | 0 to 31                                             | True     |
 * | old_in1_cb_id  | The identifier of the previously-configured srcB circular buffer (CB)  | uint32_t | 0 to 31                                             | True     |
 * | transpose      | The transpose flag for performing transpose operation on tiles in B    | uint32_t | Any positive value will indicate transpose is set   | False    |
 * | ct_dim         | The column dimension for the output block (in tiles)                   | uint32_t | 1 to 8 in half-sync mode, 1 to 16 in full-sync mode | False    |
 * | rt_dim         | The row dimension for the output block (in tiles)                      | uint32_t | 1 to 8 in half-sync mode, 1 to 16 in full-sync mode | False    |
 * | kt_dim         | The inner dimension of the input matrices (in tiles)                   | uint32_t | 1 to 2^32 - 1                                       | False    |
 */
// clang-format on
ALWI void mm_init_with_both_dt(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t old_in0_cb_id,
    uint32_t old_in1_cb_id,
    const uint32_t transpose = 0,
    uint32_t ct_dim = 1,
    uint32_t rt_dim = 1,
    uint32_t kt_dim = 1) {
#ifndef ARCH_QUASAR
    UNPACK((llk_unpack_reconfig_data_format<DST_ACCUM_MODE>(old_in1_cb_id, in1_cb_id, old_in0_cb_id, in0_cb_id)));
    MATH((llk_math_reconfig_data_format<DST_ACCUM_MODE>(old_in1_cb_id, in1_cb_id, old_in0_cb_id, in0_cb_id)));
    mm_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim);
#endif  // TODO: AM; add Quasar implementation
}

// clang-format off
/**
 * Performs tile-sized matrix multiplication *C = A * B* between the tiles in two specified
 * input CBs and accumulates the result to DST (DST += C). The DST register buffer must be
 * in acquired state via *acquire_dst* call. This call is blocking and is only available on
 * the compute engine.
 *
 * Must be preceded by a matching mm_init() call (with the same in0/in1 CBs and tile
 * defaults). For block-shaped output, use matmul_block() instead.
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB) — operand A     | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB) — operand B    | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of the tile A from the first input CB                         | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of the tile B from the second input CB                        | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG to which the result C will be written. | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
ALWI void matmul_tiles(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in0_tile_index, uint32_t in1_tile_index, uint32_t idst) {
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index)));
#ifndef ARCH_QUASAR
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst)));
#else
    MATH((llk_math_matmul_tile(idst)));
#endif
}

// clang-format off
/**
 * Performs the math half of a tile-sized matmul into DST without any unpacker work. Intended
 * for kernels that drive the unpacker manually (or have already issued a separate
 * llk_unpack_AB_matmul) and only want the math engine to compute the partial product into
 * DST. Most callers should use matmul_tiles() instead.
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | idst           | The index of the tile in DST REG to which the result C will be written. | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
template <uint32_t num_faces = 4>
ALWI void matmul_tiles_math(uint32_t idst) {
#ifndef ARCH_QUASAR
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE, num_faces>(idst)));
#endif  // TODO: AM; add Quasar implementation
}

// clang-format off
/**
 * Performs block-sized matrix multiplication *C = A * B* between the blocks in two different
 * input CBs and accumulates the result to DST (DST += C). The DST register buffer must be
 * in acquired state via *acquire_dst* call. This call is blocking and is only available on
 * the compute engine.
 *
 * Must be preceded by a matching mm_init() call configured for the same block dimensions
 * (ct_dim, rt_dim, kt_dim). For tile-shaped output, use matmul_tiles() instead.
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB) — operand A     | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB) — operand B    | uint32_t | 0 to 31                                        | True     |
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
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_block(ct_dim, rt_dim)));
#endif
}

}  // namespace ckernel
