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
    UNPACK(( llk_setup_operands() ));
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose) ));
    UNPACK(( llk_unpack_AB_matmul_hw_configure_disaggregated(in0_cb_id, in1_cb_id) ));
    #else
    UNPACK(( llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id) ));
    UNPACK(( llk_unpack_AB_matmul_hw_configure_disaggregated<DST_ACCUM_MODE>(in0_cb_id, in1_cb_id) ));
    #endif

    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(in0_cb_id, in1_cb_id, transpose) ));
    MATH(( llk_math_pack_sync_init<SYNC>()  ));
    #else
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(in0_cb_id, in1_cb_id) ));
    MATH(( llk_math_pack_sync_init<SYNC, DST_ACCUM_MODE>()  ));
    #endif

    PACK(( llk_pack_init()  ));

    #ifdef ARCH_GRAYSKULL
    PACK(( llk_pack_hw_configure_disaggregated<false>(out_cb_id) ));
    #else
    PACK(( llk_pack_hw_configure_disaggregated<false, DST_ACCUM_MODE>(out_cb_id) ));
    #endif

    PACK(( llk_setup_outputs()  ));

    #ifdef ARCH_GRAYSKULL
    PACK(( llk_pack_dest_init<SYNC, DstTileFaceLayout::RowMajor, false>()  ));
    #else
    PACK(( llk_pack_dest_init<SYNC, DstTileFaceLayout::RowMajor, false, DST_ACCUM_MODE>()  ));
    #endif

    // TODO(AP): ZM-only kernel
    PACK(( llk_init_packer_dest_offset_registers<SyncHalf,DstTileFaceLayout::RowMajor,false>()  ));
}

ALWI void mm_init_once() {

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
ALWI void matmul_tiles(uint32_t c_in0, uint32_t c_in1, uint32_t itile0, uint32_t itile1, uint32_t idst, bool transpose) {
    UNPACK(( llk_unpack_AB_matmul(c_in0,c_in1,itile0,itile1) ));
    MATH(( llk_math_matmul<MATH_FIDELITY>(idst, transpose)  ));
}

/**
 * A short version of matmul_tiles initialization.
 * It is used to reconfigure srcA of the compute engine back to matmul mode.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range                                         | Required |
 * |----------------|---------------------------------------------------------------|----------|-----------------------------------------------------|----------|
 * | cbid           | The identifier of the first input circular buffer (CB)        | uint32_t | 0 to 31                                             | True     |
 * | transpose      | The transpose flag for performing transpose operation on B    | uint32_t | Any positive value will indicate tranpose is set    | False    |
 */
ALWI void mm_init_short_with_dt(uint32_t cbid, const uint32_t transpose=0) {
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_matmul_init(cbid, 1, transpose) ));
    UNPACK(( llk_unpack_reconfig_data_format_srca(cbid, 1) ));
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(cbid, 1, transpose) ));
    #else
    UNPACK(( llk_unpack_AB_matmul_init(cbid, 1) ));
    UNPACK(( llk_unpack_reconfig_data_format_srca(cbid, 1) ));
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(cbid, 1) ));
    #endif
}

/**
 * A short version of matmul_tiles initialization.
 * Configure the unpacker and math engine to matmul mode.
 *
 * Return value: None
 *
 * | Argument       | Description                                                   | Type     | Valid Range                                         | Required |
 * |----------------|---------------------------------------------------------------|----------|-----------------------------------------------------|----------|
 * | transpose      | The transpose flag for performing transpose operation on B    | uint32_t | Any positive value will indicate tranpose is set    | False    |
 */
ALWI void mm_init_short(const std::uint32_t transpose=0) {
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(0, 1, transpose)  ));
    UNPACK(( llk_unpack_AB_matmul_init(0, 1, transpose)  ));
    #else
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(0, 1, 0)  ));
    UNPACK(( llk_unpack_AB_matmul_init(0, 1) ));
    #endif
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
 */
ALWI void mm_block_init(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, uint32_t out_cb_id = 16) {
    UNPACK(( llk_setup_operands() ));
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id) ));
    #else
    UNPACK(( llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id) ));
    #endif
    UNPACK(( llk_unpack_AB_matmul_hw_configure_disaggregated(in0_cb_id, in1_cb_id) ));

    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_matmul_init<MATH_FIDELITY, DstTileFaceLayout::ColMajor>(in0_cb_id, in1_cb_id) ));
    #else
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(in0_cb_id, in1_cb_id) ));
    #endif
    MATH(( llk_math_pack_sync_init<SYNC>()  ));

    #ifdef ARCH_GRAYSKULL
    PACK(( llk_pack_init<false, false, DstTileFaceLayout::ColMajor>()  ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(out_cb_id) ));
    PACK(( llk_setup_outputs()  ));
    PACK(( llk_pack_dest_init<SYNC, DstTileFaceLayout::ColMajor, false>()  ));
    // TODO(AP): ZM-only kernel
    PACK(( llk_init_packer_dest_offset_registers<SyncHalf,DstTileFaceLayout::ColMajor,false>()  ));
    #else
    PACK(( llk_pack_init<false, false, DstTileFaceLayout::RowMajor>()  ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(out_cb_id) ));
    PACK(( llk_setup_outputs()  ));
    PACK(( llk_pack_dest_init<SYNC, DstTileFaceLayout::RowMajor, false>()  ));
    // TODO(AP): ZM-only kernel
    PACK(( llk_init_packer_dest_offset_registers<SyncHalf,DstTileFaceLayout::RowMajor,false>()  ));
    #endif
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
 * | c_in0          | The identifier of the first input circular buffer (CB)                  | uint32_t | 0 to 31                                        | True     |
 * | c_in1          | The identifier of the second input circular buffer (CB)                 | uint32_t | 0 to 31                                        | True     |
 * | itile0         | The index of the tile in block A from the first input CB                | uint32_t | Must be less than the size of the CB           | True     |
 * | itile1         | The index of the tile in block B from the second input CB               | uint32_t | Must be less than the size of the CB           | True     |
 * | idst           | The index of the tile in DST REG to which the result C will be written. | uint32_t | Must be less than the acquired size of DST REG | True     |
 * | transpose      | The transpose flag for performing transpose operation on tiles in B.    | bool     | Must be true or false                          | True     |
 * | ct_dim         | The coloumn dimension for the output block.                             | uint32_t | Must be equal to block B column dimension      | True     |
 * | rt_dim         | The row dimension for the output block.                                 | uint32_t | Must be equal to block A row dimension         | True     |
 * | kt_dim         | The inner dimension.                                                    | uint32_t | Must be equal to block A column dimension      | True     |
 */
ALWI void matmul_block(uint32_t c_in0, uint32_t c_in1, uint32_t itile0, uint32_t itile1, uint32_t idst, bool transpose, uint32_t ct_dim, uint32_t rt_dim, uint32_t kt_dim) {
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_matmul(c_in0, c_in1, itile0, itile1, ct_dim, rt_dim, kt_dim) ));
    MATH(( llk_math_matmul<MATH_FIDELITY, DstTileFaceLayout::ColMajor>(idst, transpose, ct_dim, rt_dim, kt_dim)  ));
    #endif
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
 * | cbid           | The identifier of the output circular buffer (CB)             | uint32_t | 0 to 31                                             | False    |
 */
ALWI void mm_block_init_short_with_dt(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, uint32_t cbid=2) {
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_matmul_init(cbid, 1) ));
    UNPACK(( llk_unpack_reconfig_data_format_srca(cbid, in1_cb_id) ));
    MATH(( llk_math_matmul_init<MATH_FIDELITY, DstTileFaceLayout::ColMajor>(in0_cb_id, in1_cb_id) ));
    #else
    UNPACK(( llk_unpack_AB_matmul_init(cbid, 1) ));
    UNPACK(( llk_unpack_reconfig_data_format_srca(cbid, 1) ));
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(cbid, 1) ));
    #endif
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
 */
ALWI void mm_block_init_short(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, const std::uint32_t transpose=0) {
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_matmul_init<MATH_FIDELITY, DstTileFaceLayout::ColMajor>(in0_cb_id, in1_cb_id, transpose)  ));
    UNPACK(( llk_unpack_AB_matmul_init(0, 1, transpose)  ));
    #else
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(0, 1, 0)  ));
    UNPACK(( llk_unpack_AB_matmul_init(0, 1) ));
    #endif
}




} // namespace ckernel
