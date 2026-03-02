// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Eltwise Add Compressed kernel
//
// Computes: out[tile] = in0[tile] + in1[tile]
//
// in0: bf16 TILE_LAYOUT
// in1: bfp8 (compressed data tensor with CB format overridden to bfp8)
// out: bf16 TILE_LAYOUT

#include "../../unified_kernels/kernel_utils.hpp"

#if defined(COMPILE_FOR_TRISC)
#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
using namespace ckernel;
#elif defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#endif

void kernel_main() {
    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t num_tiles = get_named_compile_time_arg_val("num_tiles");

#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t cb_in0_num_pages = get_named_compile_time_arg_val("cb_in0_num_pages");
    constexpr uint32_t cb_in1_num_pages = get_named_compile_time_arg_val("cb_in1_num_pages");

    unified_kernels::setup_sharded_buffer(cb_in0, cb_in0_num_pages);
    unified_kernels::setup_sharded_buffer(cb_in1, cb_in1_num_pages);

#elif defined(COMPILE_FOR_BRISC)
    // BRISC: no-op

#elif defined(COMPILE_FOR_TRISC)
    deepseek_compute_kernel_init();

    reconfig_data_format<false, true>(cb_in0, cb_in1);
    pack_reconfig_data_format<true>(cb_out);
    add_tiles_init(cb_in0, cb_in1);

    cb_wait_front(cb_in0, num_tiles);
    cb_wait_front(cb_in1, num_tiles);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        add_tiles(cb_in0, cb_in1, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, 1);
    }

    cb_pop_front(cb_in0, num_tiles);
    cb_pop_front(cb_in1, num_tiles);

#endif
}
