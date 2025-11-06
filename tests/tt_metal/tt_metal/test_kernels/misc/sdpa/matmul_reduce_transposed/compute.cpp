// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_COL)
#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/softplus.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "tools/profiler/kernel_profiler.hpp"
// #include "ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"

void matmul_blocks(
    const uint32_t& in0_cb,
    const uint32_t& in1_cb,
    const uint32_t& out_cb,
    const uint32_t& M,
    const uint32_t& N,
    const uint32_t& K,
    const uint32_t& in0_num_subblocks,
    const uint32_t& in1_num_subblocks,
    const uint32_t& in0_block_w,
    const uint32_t& subblock_h,
    const uint32_t& subblock_w,
    const bool& transpose) {
    DeviceZoneScopedN("matmul_blocks");
    // precondition: in0_cb has M*K produced
    // preconditino: in1_cb has K*N produced
    // postcondition: in0_cb is full, in1_cb is empty
    // postcondition: out_cb has M*N produced
    mm_block_init_short(
        in0_cb, in1_cb, transpose /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, in0_block_w /*kt_dim*/);

    uint32_t output_num_tiles = M * N;
    uint32_t out_subblock_num_tiles = subblock_h * subblock_w;
    uint32_t in0_index_offset = 0;

    reconfig_data_format(in1_cb, in0_cb);
    pack_reconfig_data_format(out_cb);
    cb_reserve_back(out_cb, output_num_tiles);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        uint32_t in1_index_offset = 0;
        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
            tile_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                matmul_block(
                    in0_cb, in1_cb, in0_index, in1_index, dst_index, transpose, subblock_w, subblock_h, in0_block_w);
                in0_index++;
                in1_index += N;
            }
            tile_regs_commit();

            tile_regs_wait();
            uint32_t dst_idx = 0;
            uint32_t out_col_offset = in1_subblock * subblock_w;
            for (uint32_t r = 0; r < subblock_h; r++) {
                uint32_t out_row_offset = (r + subblock_h * in0_subblock) * N;
                for (uint32_t c = 0; c < subblock_w; c++) {
                    pack_tile<true>(dst_idx, out_cb, out_row_offset + out_col_offset + c);
                    dst_idx++;
                }
            }
            tile_regs_release();
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * in0_block_w;
    }
    cb_push_back(out_cb, output_num_tiles);
}

template <uint32_t in0_cb, uint32_t scale_cb, uint32_t k_chunk_size, uint32_t q_chunk_size>
void reduce_c_transposed(uint32_t out_cb) {
    DeviceZoneScopedN("reduce_c");

    constexpr uint32_t num_tiles = k_chunk_size * q_chunk_size;

    max_tile_init();
    constexpr uint32_t reduce_dst_idx = 0;
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL>(in0_cb, scale_cb, out_cb);
    for (uint32_t j = 0; j < q_chunk_size; j++) {
        acquire_dst();

        for (uint32_t i = 0; i < k_chunk_size; i++) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(
                in0_cb, scale_cb, i * q_chunk_size + j, 0, reduce_dst_idx);
        }
        pack_tile(reduce_dst_idx, out_cb);
        release_dst();
    }
    reduce_uninit();
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t k_in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t q_in_cb = get_compile_time_arg_val(1);
    constexpr uint32_t mm_out_cb = get_compile_time_arg_val(2);
    constexpr uint32_t max_out_cb = get_compile_time_arg_val(3);
    constexpr uint32_t identity_scale_cb = get_compile_time_arg_val(4);
    constexpr uint32_t k_chunk_size = get_compile_time_arg_val(5);
    constexpr uint32_t q_chunk_size = get_compile_time_arg_val(6);
    constexpr uint32_t head_dim = get_compile_time_arg_val(7);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(8);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(9);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(10);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(11);

    mm_init(k_in_cb, q_in_cb, mm_out_cb);

    matmul_blocks(
        k_in_cb,
        q_in_cb,
        mm_out_cb,
        k_chunk_size,
        q_chunk_size,
        head_dim,
        in0_num_subblocks,
        in1_num_subblocks,
        head_dim,
        subblock_h,
        subblock_w,
        true);

    // Ensure outputs are produced before exiting
    cb_wait_front(mm_out_cb, k_chunk_size * q_chunk_size);
    cb_reserve_back(max_out_cb, q_chunk_size);

    reduce_c_transposed<mm_out_cb, identity_scale_cb, k_chunk_size, q_chunk_size>(max_out_cb);

    cb_push_back(max_out_cb, q_chunk_size);
}
}  // namespace NAMESPACE
