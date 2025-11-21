// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/matmul.h"
#include "debug/dprint_tensix.h"

namespace NAMESPACE {

#ifdef TRISC_MATH
#include "llk_math_common.h"
#include "llk_math_unary_datacopy_api.h"

void math_main() {
    int __outer_loop_iter;
    llk_math_matmul_init<MATH_FIDELITY, 0>(0, 0, false, 1, 2, 1);
    llk_math_pack_sync_init<DST_ACCUM_MODE>();
    llk_math_hw_configure_disaggregated(0, 0);
    llk_math_wait_for_dest_available();
    // asm volatile("ebreak");
    llk_math_matmul<MATH_FIDELITY, 0>(0, false, 1, 2, 1);
    llk_math_dest_section_done<DST_ACCUM_MODE>();
}
#endif

#ifdef TRISC_PACK
#include "llk_pack_common.h"
#include "llk_pack.h"

void pack_main() {
    int __outer_loop_iter;
    llk_pack_hw_configure_disaggregated<DST_ACCUM_MODE, false>(16);
    llk_pack_init<false, false>(16);
    llk_pack_dest_init<DST_ACCUM_MODE, false>();
    llk_packer_wait_for_math_done();
    // asm volatile("ebreak");
    llk_wait_for_free_tiles<false, false, false>(16, 2);
    llk_pack<DST_ACCUM_MODE, false, false>(0, 16);
    llk_pack<DST_ACCUM_MODE, false, false>(1, 16);
    llk_push_tiles<false, false>(16, 2);
    // volatile uint16_t *data_ptr = reinterpret_cast<uint16_t*>(get_local_cb_interface(16).fifo_wr_ptr << 4);
    // const char* digits = "0123456789ABCDEF";
    // DPRINT << "out data_ptr: " << data_ptr << ENDL();
    // for (int i = 0; i < (int)2; i++) {
    //     for (int r = 0; r < 34; r++) {
    //         for (int c = 0; c < 16; c++) {
    //             DPRINT << digits[(data_ptr[i*34*16+r*16+c] & 0xf000)>>12]
    //                    << digits[(data_ptr[i*34*16+r*16+c] & 0x0f00)>>8]
    //                    << digits[(data_ptr[i*34*16+r*16+c] & 0x00f0)>>4]
    //                    << digits[(data_ptr[i*34*16+r*16+c] & 0x000f)>>0] << " ";
    //         }
    //         DPRINT << ENDL();
    //     }
    //     DPRINT << ENDL();
    // }
    // llk_pack_dest_section_done<DST_ACCUM_MODE>();
}
#endif

#ifdef TRISC_UNPACK
void unpack_main() {
    int __outer_loop_iter;
    llk_unpack_AB_matmul_hw_configure_disaggregated<DST_ACCUM_MODE>(0, 0);
    llk_unpack_AB_matmul_init(0, 0, false, 1, 2, 1);
    llk_wait_tiles(0, 2);
    // asm volatile("ebreak");
    // volatile uint16_t *data_ptr = reinterpret_cast<uint16_t*>(get_local_cb_interface(0).fifo_rd_ptr << 4);
    // const char* digits = "0123456789ABCDEF";
    // DPRINT << "data_ptr: " << data_ptr << ENDL();
    // for (int i = 0; i < (int)2; i++) {
    //     for (int r = 0; r < 34; r++) {
    //         for (int c = 0; c < 16; c++) {
    //             DPRINT << digits[(data_ptr[i*34*16+r*16+c] & 0xf000)>>12]
    //                    << digits[(data_ptr[i*34*16+r*16+c] & 0x0f00)>>8]
    //                    << digits[(data_ptr[i*34*16+r*16+c] & 0x00f0)>>4]
    //                    << digits[(data_ptr[i*34*16+r*16+c] & 0x000f)>>0] << " ";
    //         }
    //         DPRINT << ENDL();
    //     }
    //     DPRINT << ENDL();
    // }
    // llk_unpack_A(0, 0);
    llk_unpack_AB_matmul(0, 0, 0, 0, 1, 2, 1);
    llk_pop_tiles(0, 2);
}
#endif

}  // namespace NAMESPACE
