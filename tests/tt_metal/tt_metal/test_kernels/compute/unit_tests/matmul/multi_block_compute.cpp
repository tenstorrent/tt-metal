// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t in0_block_num_tiles = get_arg(args::in0_block_num_tiles);
    constexpr uint32_t in1_block_num_tiles = get_arg(args::in1_block_num_tiles);
    constexpr uint32_t out_block_num_tiles = get_arg(args::out_block_num_tiles);
    constexpr uint32_t out_r = get_arg(args::out_r);
    constexpr uint32_t out_c = get_arg(args::out_c);
    constexpr uint32_t in0_k = get_arg(args::in0_k);
    constexpr uint32_t num_blocks = get_arg(args::num_blocks);
    constexpr uint32_t last_block_id = num_blocks - 1;

    // Buffers arrive through host-declared bindings rather than compile-time ids. DataflowBuffer
    // is arch-agnostic under the 2.0 host API (it maps onto CBs on Gen1), so the previous
    // #ifdef ARCH_QUASAR CircularBuffer fork and its get_buffer_id shim are gone.
    DataflowBuffer cb_in0(dfb::in0);
    DataflowBuffer cb_in1(dfb::in1);
    DataflowBuffer cb_partials(dfb::partials);
    DataflowBuffer cb_out(dfb::out);
    const uint32_t in0_id = cb_in0.get_id();
    const uint32_t in1_id = cb_in1.get_id();
    const uint32_t out_id = cb_out.get_id();
    const uint32_t partials_id = cb_partials.get_id();

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
