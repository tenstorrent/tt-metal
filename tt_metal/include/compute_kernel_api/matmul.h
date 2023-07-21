#pragma once


#include "compute_kernel_api/llk_matmul_includes.h"
#include "compute_kernel_api/llk_unpack_AB_matmul_includes.h"


namespace ckernel {

ALWI void mm_init(uint32_t in0_cb_id = 0, uint32_t in1_cb_id = 1, uint32_t out_cb_id = 16) {
    UNPACK(( llk_setup_operands() ));
    UNPACK(( llk_unpack_AB_matmul_init() ));
    UNPACK(( llk_unpack_AB_matmul_hw_configure_disaggregated(in0_cb_id, in1_cb_id) ));

    MATH(( llk_math_matmul_init<MATH_FIDELITY>() ));
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
    MATH(( llk_math_matmul<MATH_FIDELITY>(idst)  ));
}

ALWI void mm_init_short_with_dt(uint32_t cbid) {
    UNPACK(( llk_unpack_AB_matmul_init() ));
    UNPACK(( llk_unpack_reconfig_data_format(cbid, 1, 0, 0) ));
    MATH(( llk_math_matmul_init<MATH_FIDELITY>() ));
}

ALWI void mm_init_short() {
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(0)  ));

    UNPACK(( llk_unpack_AB_matmul_init(0)  ));
}



} // namespace ckernel
