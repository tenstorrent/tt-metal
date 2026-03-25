// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/transpose_wh.h"
#include "experimental/circular_buffer.h"

// DeepSeek Top32 headers (repo-relative; JIT adds -I only for this file's directory).
#if defined(TRISC_UNPACK)
#include "../../../../../models/demos/deepseek_v3_b1/kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_A_top32_rm_api.h"
#endif

#if defined(TRISC_MATH)
#include "../../../../../models/demos/deepseek_v3_b1/kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_deepseek_top32_rm.h"
#include "../../../../../models/demos/deepseek_v3_b1/kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_top32_rm_api.h"
#endif

void kernel_main() {
    const uint32_t value_offset_tiles = 0;
    const uint32_t index_offset_tiles = 2;
    const uint32_t row_elements = get_compile_time_arg_val(0);
    const uint32_t num_input_tiles = get_compile_time_arg_val(1);
    const uint32_t num_output_tiles = get_compile_time_arg_val(2);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t cb_out1 = tt::CBIndex::c_17;

    experimental::CircularBuffer cb0(cb_in0);
    experimental::CircularBuffer cb1(cb_in1);
    experimental::CircularBuffer cb16(cb_out0);
    experimental::CircularBuffer cb17(cb_out1);

    ckernel::compute_kernel_hw_startup(cb_in0, cb_in1, cb_out0);

    cb0.wait_front(num_input_tiles);
    cb1.wait_front(num_input_tiles);

    cb16.reserve_back(num_output_tiles);
    cb17.reserve_back(num_output_tiles);

    acquire_dst();

    /*
    Algorithm implementation:
    1. unpack first 64 elements from in0 and in1 into Dest
        - unpack 16 elements at a time into 1st row of each face in SRCA with transpose (ends up in 1st col)
        - this can fit up to 64 elements at a time (4faces x 16rows)
        - initialize the rest to -inf so that they dont get in the way of the sort (can work with less than 64 elements)
    2. sort in decreasing order using bitonic sort to get top32 tile sort started
    loop for number of remaining values:
        3. unpack up to 64 more elements same way as step 1
        4. sort in increasing order using bitonic sort to get next top 32
        5. bitonic merge the top32 array + rebuild the top32 array
    6. transpose back and pack the final top32 array into out0 and indices into out1
    */

    // step 1
    uint32_t num_faces = 4;
    reconfig_data_format_srca(cb_in0);
    UNPACK((llk_unpack_A_top32_rm_init(cb_in0)));
    UNPACK((llk_unpack_A_top32_rm(cb_in0, 0, num_faces)));
    MATH((llk_math_top32_rm_init(cb_in0)));
    MATH((llk_math_top32_rm(cb_in0, value_offset_tiles, num_faces)));

    reconfig_data_format_srca(cb_in1);
    UNPACK((llk_unpack_A_top32_rm_init(cb_in1)));
    UNPACK((llk_unpack_A_top32_rm(cb_in1, 0, num_faces)));
    MATH((llk_math_top32_rm_init(cb_in1)));
    MATH((llk_math_top32_rm(cb_in1, index_offset_tiles, num_faces)));

    // step 2
    uint32_t decreasing = 0;
    uint32_t increasing = 1;
    MATH((llk_math_deepseek_top32_rm_init<false>()));
    MATH((llk_math_deepseek_top32_rm_local_sort<false, DST_ACCUM_MODE>(value_offset_tiles, decreasing)));
    MATH((llk_math_deepseek_top32_rm_merge<false, DST_ACCUM_MODE>(value_offset_tiles, /*across_tiles*/ false)));
    MATH((llk_math_deepseek_top32_rm_rebuild<false, DST_ACCUM_MODE>(
        value_offset_tiles, decreasing, /*skip_second*/ true)));

    // loop for number of remaining values:
    for (uint32_t i = 64; i < row_elements; i += 64) {
        if (i + 64 > row_elements) {
            // process just 32 elements
            num_faces = 2;
        } else {
            // process full 64 elements
            num_faces = 4;
        }

        // step 3
        reconfig_data_format_srca(cb_in0);
        UNPACK((llk_unpack_A_top32_rm_init(cb_in0)));
        UNPACK((llk_unpack_A_top32_rm(cb_in0, i / 64, num_faces)));
        MATH((llk_math_top32_rm_init(cb_in0)));
        MATH((llk_math_top32_rm(cb_in0, value_offset_tiles + 1, num_faces)));

        reconfig_data_format_srca(cb_in1);
        UNPACK((llk_unpack_A_top32_rm_init(cb_in1)));
        UNPACK((llk_unpack_A_top32_rm(cb_in1, i / 64, num_faces)));
        MATH((llk_math_top32_rm_init(cb_in1)));
        MATH((llk_math_top32_rm(cb_in1, index_offset_tiles + 1, num_faces)));

        // step 4
        MATH((llk_math_deepseek_top32_rm_local_sort<false, DST_ACCUM_MODE>(value_offset_tiles + 1, decreasing)));
        MATH((llk_math_deepseek_top32_rm_merge<false, DST_ACCUM_MODE>(value_offset_tiles + 1, /*across_tiles*/ false)));
        MATH((llk_math_deepseek_top32_rm_rebuild<false, DST_ACCUM_MODE>(
            value_offset_tiles + 1, increasing, /*skip_second*/ true)));

        // step 5
        MATH((llk_math_deepseek_top32_rm_merge<false, DST_ACCUM_MODE>(value_offset_tiles, /*across_tiles*/ true)));
        MATH((llk_math_deepseek_top32_rm_rebuild<false, DST_ACCUM_MODE>(
            value_offset_tiles, decreasing, /*skip_second*/ true)));
    }

    //     tensix_sync();
    //     // use risc to read DEST directly
    //     cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_fmt_RMW, 0b001 /* RISC_DEST_FMT_INT32 */);
    //     cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_no_swizzle_RMW, 0);
    //     cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_unsigned_int_RMW, 0);
    //     tensix_sync();

    // #if defined(TRISC_MATH)
    //     volatile uint32_t* dst32 = reinterpret_cast<volatile uint32_t*>(0xFFBD8000U);
    //     DPRINT << "DEST[row][col] raw INT32 (hex):" << ENDL();
    //     for (int row = 0; row < 4*64; row++) {
    //         for (int col = 0; col < 16; col++) {
    //             uint32_t val = dst32[row * 16 + col];
    //             float f;
    //             std::memcpy(&f, &val, sizeof(f));
    //             DPRINT << f << " ";
    //             // DPRINT << HEX() << val << " ";
    //         }
    //         DPRINT << ENDL();
    //     }
    // #endif

    // step 6
    PACK(TTI_SETADCXX(p_setadc::PAC, 1 - 1, 0x0));
    ckernel::pack_tile(value_offset_tiles, cb_out0);
    ckernel::pack_reconfig_data_format(cb_out0, cb_out1);
    ckernel::pack_tile(index_offset_tiles, cb_out1);

    release_dst();

    cb0.pop_front(num_input_tiles);
    cb1.pop_front(num_input_tiles);

    cb16.push_back(num_output_tiles);
    cb17.push_back(num_output_tiles);
}
