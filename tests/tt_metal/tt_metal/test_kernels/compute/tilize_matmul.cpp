// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    const uint32_t in0_cb = get_compile_time_arg_val(0);
    const uint32_t in1_cb = get_compile_time_arg_val(1);
    const uint32_t imm_cb = get_compile_time_arg_val(2);
    const uint32_t out_cb = get_compile_time_arg_val(3);
    const uint32_t num_in0_tiles = 1;
    const uint32_t num_in1_tiles = 1;
    const uint32_t num_out_tiles = 1;
    const uint32_t in0_tile_index = 0;
    const uint32_t in1_tile_index = 0;
    const uint32_t out_tile_index = 0;
    const bool transpose = false;

    cb_reserve_back(3, 1);
    volatile std::uint32_t* base_address = (std::uint32_t*)6768;

    cb_reserve_back(imm_cb, num_out_tiles);
    cb_wait_front(in0_cb, num_in0_tiles);

    tilize_init(in0_cb, num_in0_tiles, imm_cb);

    tensix_sync();
    UNPACK((base_address[0] = 1));
    MATH((base_address[1] = 2));
    PACK((base_address[2] = 3));
    while (base_address[0] != 1) {
        asm("nop");
    }
    while (base_address[1] != 2) {
        asm("nop");
    }
    while (base_address[2] != 3) {
        asm("nop");
    }
    UNPACK((base_address[0] = 0));
    MATH((base_address[1] = 0));
    PACK((base_address[2] = 0));
    while (base_address[0] != 0) {
        asm("nop");
    }
    while (base_address[1] != 0) {
        asm("nop");
    }
    while (base_address[2] != 0) {
        asm("nop");
    }

    tilize_block(in0_cb, num_in0_tiles, imm_cb);

    tensix_sync();

    tilize_uninit(in0_cb, imm_cb);

    cb_pop_front(in0_cb, num_in0_tiles);
    cb_push_back(imm_cb, num_in0_tiles);

    cb_reserve_back(out_cb, num_out_tiles);
    cb_wait_front(in1_cb, num_in1_tiles);
    cb_wait_front(imm_cb, num_in1_tiles);

    mm_init(imm_cb, in1_cb, out_cb);

    UNPACK((base_address[0] = 1));
    MATH((base_address[1] = 2));
    PACK((base_address[2] = 3));
    while (base_address[0] != 1) {
        asm("nop");
    }
    while (base_address[1] != 2) {
        asm("nop");
    }
    while (base_address[2] != 3) {
        asm("nop");
    }
    UNPACK((base_address[0] = 0));
    MATH((base_address[1] = 0));
    PACK((base_address[2] = 0));
    while (base_address[0] != 0) {
        asm("nop");
    }
    while (base_address[1] != 0) {
        asm("nop");
    }
    while (base_address[2] != 0) {
        asm("nop");
    }

    acquire_dst();
    matmul_tiles(imm_cb, in1_cb, in0_tile_index, in1_tile_index, out_tile_index, transpose);
    pack_tile(0, out_cb);
    release_dst();

    tensix_sync();

    cb_pop_front(in1_cb, num_in1_tiles);
    cb_pop_front(imm_cb, num_in1_tiles);
    cb_push_back(out_cb, num_out_tiles);
}
}  // namespace NAMESPACE
