// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/eltwise_binary.h"
#include "api/debug/dprint_tensix.h"
#include "api/dataflow/circular_buffer.h"

// #include "api/debug/dprint.h"
inline void tilizeA_B_binary_init(uint32_t icb0, uint32_t icb1, uint32_t block) {
    UNPACK((llk_unpack_tilizeA_B_init<true, true>(icb0, icb1, block)));

    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWADD, BroadcastType::NONE, MathFidelity::LoFi>(
        icb0, icb1, 0 /*acc_to_dest*/)));
}

inline void add_tiles_math(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<
        EltwiseBinaryType::ELWADD,
        BroadcastType::NONE,
        DST_ACCUM_MODE,
        MathFidelity::LoFi,
        EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true /*clear_fp32_dst_acc*/)));
}

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb1(tt::CBIndex::c_1);
    CircularBuffer cb16(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    tilizeA_B_binary_init(tt::CBIndex::c_0, tt::CBIndex::c_1, per_core_block_tile_cnt);

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        cb0.wait_front(per_core_block_tile_cnt);
        cb1.wait_front(per_core_block_tile_cnt);
        cb16.reserve_back(per_core_block_tile_cnt);
        unpack_tilizeA_B_block(tt::CBIndex::c_0, tt::CBIndex::c_1, per_core_block_tile_cnt, b);

        for (uint i = 0; i < per_core_block_tile_cnt; ++i) {
            tile_regs_acquire();
            add_tiles_math(tt::CBIndex::c_0, tt::CBIndex::c_1, i, i, 0);
            // dprint_tensix_dest_reg(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, tt::CBIndex::c_16);
            tile_regs_release();
        }

        cb16.push_back(per_core_block_tile_cnt);
        cb0.pop_front(per_core_block_tile_cnt);
        cb1.pop_front(per_core_block_tile_cnt);
    }
}
