

// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#ifdef ARCH_QUASAR
#include "api/dataflow/dataflow_buffer.h"
#else
#include "api/dataflow/circular_buffer.h"
#endif

#ifdef ARCH_QUASAR
using Buffer = DataflowBuffer;
inline uint32_t get_buffer_id(const Buffer& b) { return b.get_id(); }
#else
using Buffer = CircularBuffer;
inline uint32_t get_buffer_id(const Buffer& b) { return b.get_cb_id(); }
#endif

void kernel_main() {
    const uint32_t in0_cb = get_compile_time_arg_val(0);
    const uint32_t in1_cb = get_compile_time_arg_val(1);
    const uint32_t out_cb = get_compile_time_arg_val(2);
    const uint32_t partials_cb = get_compile_time_arg_val(3);
    const uint32_t in0_block_num_tiles = get_compile_time_arg_val(4);
    const uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);
    const uint32_t out_block_num_tiles = get_compile_time_arg_val(6);
    const uint32_t out_r = get_compile_time_arg_val(7);
    const uint32_t out_c = get_compile_time_arg_val(8);
    const uint32_t in0_k = get_compile_time_arg_val(9);
    const uint32_t num_blocks = get_compile_time_arg_val(10);
    const uint32_t last_block_id = num_blocks - 1;

    Buffer cb_in0(in0_cb);
    Buffer cb_in1(in1_cb);
    Buffer cb_partials(partials_cb);
    Buffer cb_out(out_cb);
    const uint32_t in0_id = get_buffer_id(cb_in0);
    const uint32_t in1_id = get_buffer_id(cb_in1);
    const uint32_t out_id = get_buffer_id(cb_out);
    const uint32_t partials_id = get_buffer_id(cb_partials);

    // out = in0[r x k]*in1[k x c]
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_id, in1_id, partials_id);
    matmul_init(in0_id, in1_id);

    for (uint32_t block_id = 0; block_id < num_blocks; block_id++) {
        tile_regs_acquire();
#ifndef PACKER_L1_ACC
        if (block_id > 0) {
            copy_init(partials_id);
            cb_partials.wait_front(out_block_num_tiles);
            for (uint32_t i = 0; i < out_block_num_tiles; i++) {
                copy_tile(partials_id, i, i);
            }
            cb_partials.pop_front(out_block_num_tiles);
            matmul_init(in0_id, in1_id);
        }
#endif

        uint32_t out_tile_index = 0;
        uint32_t in0_index_r_offset = 0;
        cb_in0.wait_front(in0_block_num_tiles);
        cb_in1.wait_front(in1_block_num_tiles);
        for (uint32_t r = 0; r < out_r; r++) {
            for (uint32_t c = 0; c < out_c; c++) {
                uint32_t in1_index_c_offset = 0;
                for (uint32_t k = 0; k < in0_k; k++) {
                    int in0_tile_index = in0_index_r_offset + k;
                    int in1_tile_index = in1_index_c_offset + c;
                    matmul_tiles(in0_id, in1_id, in0_tile_index, in1_tile_index, out_tile_index);
                    in1_index_c_offset += out_c;
                }
                out_tile_index++;
            }
            in0_index_r_offset += in0_k;
        }
        cb_in0.pop_front(in0_block_num_tiles);
        cb_in1.pop_front(in1_block_num_tiles);

        tile_regs_commit();
        tile_regs_wait();

#ifdef PACKER_L1_ACC
        cb_partials.reserve_back(out_block_num_tiles);
        if (block_id == 0) {
            pack_reconfig_l1_acc(0);
        }
        for (uint32_t tile_index = 0; tile_index < out_block_num_tiles; tile_index++) {
            pack_tile(tile_index, partials_id);
        }
        cb_partials.push_back(out_block_num_tiles);
        if (block_id == 0) {
            pack_reconfig_l1_acc(1);
        }
        tile_regs_release();
        if (block_id < last_block_id) {
            cb_partials.wait_front(out_block_num_tiles);
            cb_partials.pop_front(out_block_num_tiles);
        }
#else
        const bool is_last = (block_id == last_block_id);
        auto& cb_dst = is_last ? cb_out : cb_partials;
        const uint32_t cb_dst_id = is_last ? out_id : partials_id;
        if (is_last) {
            pack_init(out_id);
        }
        cb_dst.reserve_back(out_block_num_tiles);
        for (uint32_t tile_index = 0; tile_index < out_block_num_tiles; tile_index++) {
            pack_tile(tile_index, cb_dst_id);
        }
        cb_dst.push_back(out_block_num_tiles);
        tile_regs_release();
#endif
    }

#ifdef PACKER_L1_ACC
    pack_reconfig_l1_acc(0);

    copy_init(partials_id);
    cb_partials.wait_front(out_block_num_tiles);
    tile_regs_acquire();
    for (uint32_t i = 0; i < out_block_num_tiles; i++) {
        copy_tile(partials_id, i, i);
    }
    cb_partials.pop_front(out_block_num_tiles);

    tile_regs_commit();
    tile_regs_wait();

    pack_init(out_id);
    cb_out.reserve_back(out_block_num_tiles);
    for (uint32_t tile_index = 0; tile_index < out_block_num_tiles; tile_index++) {
        pack_tile(tile_index, out_id);
    }
    cb_out.push_back(out_block_num_tiles);
    tile_regs_release();
#endif
}
