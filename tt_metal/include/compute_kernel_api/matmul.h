// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_matmul_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_matmul_api.h"
#endif

namespace ckernel {

/**
 * Initialization for matmul_tiles operation. Must be called before matmul_tiles.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range                                         | Required |
 * |----------------|---------------------------------------------------------------|----------|-----------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)        | uint32_t | 0 to 31                                             | False    |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)       | uint32_t | 0 to 31                                             | False    |
 * | out_cb_id      | The identifier of the output circular buffer (CB)             | uint32_t | 0 to 31                                             | False    |
 * | transpose      | The transpose flag for performing transpose operation on B    | uint32_t |  Any positive value will indicate tranpose is set   | False    |
 */
ALWI void mm_init(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, uint32_t out_cb_id = 16, const uint32_t transpose=0) {
    UNPACK(( llk_unpack_AB_matmul_hw_configure_disaggregated<DST_ACCUM_MODE>(in0_cb_id, in1_cb_id) ));
    UNPACK(( llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose) ));

    MATH(( llk_math_matmul_init<MATH_FIDELITY>(in0_cb_id, in1_cb_id, transpose) ));
    MATH(( llk_math_pack_sync_init<DST_ACCUM_MODE>()  ));
    MATH(( llk_math_hw_configure_disaggregated(in0_cb_id, in1_cb_id) ));

    PACK(( llk_pack_hw_configure_disaggregated<false, DST_ACCUM_MODE>(out_cb_id) ));
    PACK(( llk_pack_init(out_cb_id)  ));
    PACK(( llk_pack_dest_init<false, DST_ACCUM_MODE>()  ));
}

/**
 * Performs tile-sized matrix multiplication *C=A\*B* between the tiles in two
 * specified input CBs and writes the result to DST. The DST register buffer
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
ALWI void matmul_tiles(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in0_tile_index, uint32_t in1_tile_index, uint32_t idst, const uint32_t transpose) {
    UNPACK(( llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index) ));
    MATH(( llk_math_matmul<MATH_FIDELITY>(idst, transpose)  ));
}

/**
 * Performs tile-sized matrix multiplication *C=A\*B* between the tiles
 * located in SRCA and SRCB and writes the result to DST. The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking and
 * is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | idst           | The index of the tile in DST REG to which the result C will be written. | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
template <uint32_t num_faces = 4>
ALWI void matmul_tiles_math(uint32_t idst) {
    MATH(( llk_math_matmul<MATH_FIDELITY, num_faces>(idst)  ));
}

/**
 * A short version of matmul_tiles initialization.
 * Configure the unpacker and math engine to matmul mode.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range                                         | Required |
 * |----------------|---------------------------------------------------------------|----------|-----------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)        | uint32_t | 0 to 31                                             | False    |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)       | uint32_t | 0 to 31                                             | False    |
 * | transpose      | The transpose flag for performing transpose operation on B    | uint32_t | Any positive value will indicate tranpose is set    | False    |
 */
ALWI void mm_init_short(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, const uint32_t transpose=0) {
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(in0_cb_id, in1_cb_id, transpose)  ));
    UNPACK(( llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose)  ));
}

/**
 * A short version of matmul_tiles initialization.
 * It is used to reconfigure srcA of the compute engine back to matmul mode.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range                                         | Required |
 * |----------------|---------------------------------------------------------------|----------|-----------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)        | uint32_t | 0 to 31                                             | False    |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)       | uint32_t | 0 to 31                                             | False    |
 * | c_in_old_srca  | The identifier of the old input to src A circular buffer (CB) | uint32_t | 0 to 31                                             | False    |
 * | transpose      | The transpose flag for performing transpose operation on B    | uint32_t | Any positive value will indicate tranpose is set    | False    |
 */
ALWI void mm_init_short_with_dt(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, uint32_t c_in_old_srca = 2, const uint32_t transpose=0) {
    UNPACK(( llk_unpack_reconfig_data_format_srca(c_in_old_srca, in1_cb_id) ));
    MATH(( llk_math_reconfig_data_format_srca(c_in_old_srca, in1_cb_id) ));
    mm_init_short(in0_cb_id, in1_cb_id, transpose);
}

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
 * | ct_dim         | the number of columns of the output matrix in tiles           | uint32_t | 1 to 8 in half-sync mode, 1 to 16 in full-sync mode | False    |
 * | rt_dim         | the number of rows of the output matrix in tiles              | uint32_t | 1 to 8 in half-sync mode, 1 to 16 in full-sync mode | False    |
 * | kt_dim         | the inner dim of the input matrices in tiles                  | uint32_t | 1 to 2^32-1                                         | False    |
 */
ALWI void mm_block_init(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, uint32_t out_cb_id = 16, const uint32_t transpose = 0, uint32_t ct_dim = 1, uint32_t rt_dim = 1, uint32_t kt_dim = 1) {
    UNPACK(( llk_unpack_AB_matmul_hw_configure_disaggregated<DST_ACCUM_MODE>(in0_cb_id, in1_cb_id) ));
    UNPACK(( llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim) ));

    MATH(( llk_math_matmul_init<MATH_FIDELITY>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim) ));
    MATH(( llk_math_pack_sync_init<DST_ACCUM_MODE>()  ));

    PACK(( llk_pack_hw_configure_disaggregated<false, DST_ACCUM_MODE>(out_cb_id) ));
    PACK(( llk_pack_init<false, false>(out_cb_id)  ));
    PACK(( llk_pack_dest_init<false, DST_ACCUM_MODE>()  ));
}

/**
 * Performs block-sized matrix multiplication *C=A\*B* between the blocks in two
 * different input CBs and writes the result to DST. The DST register buffer
 * must be in acquired state via *acquire_dst* call. This call is blocking and
 * is only available on the compute engine.
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
 * | ct_dim         | The coloumn dimension for the output block.                             | uint32_t | Must be equal to block B column dimension      | True     |
 * | rt_dim         | The row dimension for the output block.                                 | uint32_t | Must be equal to block A row dimension         | True     |
 * | kt_dim         | The inner dimension.                                                    | uint32_t | Must be equal to block A column dimension      | True     |
 */
ALWI void matmul_block(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in0_tile_index, uint32_t in1_tile_index, uint32_t idst, const uint32_t transpose, uint32_t ct_dim, uint32_t rt_dim, uint32_t kt_dim) {
    UNPACK(( llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim) ));
    MATH(( llk_math_matmul<MATH_FIDELITY>(idst, transpose, ct_dim, rt_dim, kt_dim)  ));
}

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
 * | transpose      | The transpose flag for performing transpose operation on B    | uint32_t | Any positive value will indicate tranpose is set    | False    |
 * | ct_dim         | The coloumn dimension for the output block.                   | uint32_t | Must be equal to block B column dimension           | False    |
 * | rt_dim         | The row dimension for the output block.                       | uint32_t | Must be equal to block A row dimension              | False    |
 * | kt_dim         | The inner dimension.                                          | uint32_t | Must be equal to block A column dimension           | False    |
 */
ALWI void mm_block_init_short(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, const uint32_t transpose=0, uint32_t ct_dim = 1, uint32_t rt_dim = 1, uint32_t kt_dim = 1) {
    UNPACK(( llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim) ));
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)  ));
}

/**
 * A short version of matmul_block initialization.
 * It is used to reconfigure srcA of the compute engine back to matmul mode.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range                                         | Required |
 * |----------------|---------------------------------------------------------------|----------|-----------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the first input circular buffer (CB)        | uint32_t | 0 to 31                                             | False    |
 * | in1_cb_id      | The identifier of the second input circular buffer (CB)       | uint32_t | 0 to 31                                             | False    |
 * | old_in1_cb_id  | The identifier of the old in1_cb_id circular buffer (CB)      | uint32_t | 0 to 31                                             | False    |
 * | ct_dim         | The coloumn dimension for the output block.                   | uint32_t | Must be equal to block B column dimension           | False    |
 * | rt_dim         | The row dimension for the output block.                       | uint32_t | Must be equal to block A row dimension              | False    |
 * | kt_dim         | The inner dimension.                                          | uint32_t | Must be equal to block A column dimension           | False    |
 */
ALWI void mm_block_init_short_with_dt(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, uint32_t old_in1_cb_id=2, const uint32_t transpose=0, uint32_t ct_dim = 1, uint32_t rt_dim = 1, uint32_t kt_dim = 1) {
    UNPACK(( llk_unpack_reconfig_data_format_srca(old_in1_cb_id, in1_cb_id) ));
    MATH(( llk_math_reconfig_data_format_srca(old_in1_cb_id, in1_cb_id) ));
    mm_block_init_short(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim);
}



} // namespace ckernel
