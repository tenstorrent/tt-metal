// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_matmul.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_matmul.h"
#endif

namespace ckernel {

ALWI void mm_init(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, uint32_t out_cb_id = 16, const uint32_t transpose=0) {
    UNPACK(( llk_setup_operands() ));
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_matmul_init(transpose) ));
    #else
    UNPACK(( llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id) ));
    #endif
    UNPACK(( llk_unpack_AB_matmul_hw_configure_disaggregated(in0_cb_id, in1_cb_id) ));

    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(transpose) ));
    #else
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(in0_cb_id, in1_cb_id) ));
    #endif
    MATH(( llk_math_pack_sync_init<SYNC>()  ));

    PACK(( llk_pack_init()  ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(out_cb_id) ));
    PACK(( llk_setup_outputs()  ));
    PACK(( llk_pack_dest_init<SYNC, DstTileFaceLayout::RowMajor, false>()  ));
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

ALWI void mm_init_short_with_dt(uint32_t cbid, const uint32_t transpose=0) {
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_matmul_init(transpose) ));
    UNPACK(( llk_unpack_reconfig_data_format_srca(cbid, 1) ));
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(transpose) ));
    #else
    UNPACK(( llk_unpack_AB_matmul_init(cbid, 1) ));
    UNPACK(( llk_unpack_reconfig_data_format_srca(cbid, 1) ));
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(cbid, 1) ));
    #endif
}

ALWI void mm_init_short(const std::uint32_t transpose=0) {
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(transpose)  ));
    UNPACK(( llk_unpack_AB_matmul_init(transpose)  ));
    #else
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(0, 1, 0)  ));
    UNPACK(( llk_unpack_AB_matmul_init(0, 1) ));
    #endif
}


ALWI void mm_block_init(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, uint32_t out_cb_id = 16) {
    UNPACK(( llk_setup_operands() ));
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_matmul_init_cm<false>(in0_cb_id, in1_cb_id) ));
    #else
    UNPACK(( llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id) ));
    #endif
    UNPACK(( llk_unpack_AB_matmul_hw_configure_disaggregated(in0_cb_id, in1_cb_id) ));

    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_matmul_init_cm<MATH_FIDELITY, DstTileFaceLayout::ColMajor, false>(in0_cb_id, in1_cb_id) ));
    #else
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(in0_cb_id, in1_cb_id) ));
    #endif
    MATH(( llk_math_pack_sync_init<SYNC>()  ));

    PACK(( llk_pack_init<false, false, DstTileFaceLayout::ColMajor>()  ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(out_cb_id) ));
    PACK(( llk_setup_outputs()  ));
    PACK(( llk_pack_dest_init<SYNC, DstTileFaceLayout::ColMajor, false>()  ));
    // TODO(AP): ZM-only kernel
    PACK(( llk_init_packer_dest_offset_registers<SyncHalf,DstTileFaceLayout::ColMajor,false>()  ));
}

ALWI void matmul_block(uint32_t c_in0, uint32_t c_in1, uint32_t itile0, uint32_t itile1, uint32_t idst, bool transpose, uint32_t ct_dim, uint32_t rt_dim, uint32_t kt_dim) {
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_matmul_cm(c_in0, c_in1, itile0, itile1, ct_dim, rt_dim, kt_dim) ));
    MATH(( llk_math_matmul_cm<MATH_FIDELITY, DstTileFaceLayout::ColMajor>(idst, transpose, ct_dim, rt_dim, kt_dim)  ));
    #endif
}


ALWI void mm_block_init_short_with_dt(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, uint32_t cbid=2) {
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_matmul_init_cm<false>(in0_cb_id, in1_cb_id) ));
    UNPACK(( llk_unpack_reconfig_data_format_srca(cbid, in1_cb_id) ));
    MATH(( llk_math_matmul_init_cm<MATH_FIDELITY, DstTileFaceLayout::ColMajor, false>(in0_cb_id, in1_cb_id) ));
    #else
    UNPACK(( llk_unpack_AB_matmul_init(cbid, 1) ));
    UNPACK(( llk_unpack_reconfig_data_format_srca(cbid, 1) ));
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(cbid, 1) ));
    #endif
}

ALWI void mm_block_init_short(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, const std::uint32_t transpose=0) {
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_matmul_init_cm<MATH_FIDELITY>(in0_cb_id, in1_cb_id, transpose)  ));
    UNPACK(( llk_unpack_AB_matmul_init_cm(in0_cb_id, in1_cb_id, transpose)  ));
    #else
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(0, 1, 0)  ));
    UNPACK(( llk_unpack_AB_matmul_init(0, 1) ));
    #endif
}




} // namespace ckernel
