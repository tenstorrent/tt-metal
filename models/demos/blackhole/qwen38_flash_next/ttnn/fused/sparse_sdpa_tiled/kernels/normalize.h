// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The final normalization of the streaming flash loop with the operand formats set explicitly: compute_streaming's
// normalize_row_streaming reconfigures the unpacker for the col-identity / scratch pair before its matmul and then
// multiplies the running out by the reciprocal with the unpacker's srcA still on the col-identity's format, which
// is right only when out, scratch and col-identity share one format (bf16 everywhere).  With an fp32 running state
// (out, sum, scratch fp32; col-identity bf16) that read returned garbage (zeros / inf on device, 2026-09-25), so the
// kernel carries this copy: per tile row, row sum -> matmul against the col-identity -> reciprocal into the scratch
// CB -> out * recip(sum) packed to the bf16 output CB.  Same arithmetic, same primitives, same pops.

#pragma once

namespace sst {

template <
    uint32_t head_dim_t_,
    uint32_t dst_size,
    uint32_t col_identity_cb,
    uint32_t scratch_cb,
    uint32_t normalized_out_cb>
static __attribute__((noinline, noclone)) void normalize_rows(uint32_t cur_sum_cb, uint32_t cur_out_cb, uint32_t sbh) {
    for (uint32_t s = 0; s < sbh; s++) {
        {
            // 1 + 2: sum x col_identity (matmul: in0 = sum -> srcB, in1 = col_identity -> srcA) -> recip -> scratch
            constexpr uint32_t N = 1;
            reconfig_data_format(col_identity_cb, cur_sum_cb);
            pack_reconfig_data_format(scratch_cb);
            matmul_block_init(cur_sum_cb, col_identity_cb, 0, N, 1, N);
            CircularBuffer(col_identity_cb).wait_front(N);
            CircularBuffer(cur_sum_cb).wait_front(1);
            CircularBuffer(scratch_cb).reserve_back(1);
            tile_regs_acquire();
            matmul_block(cur_sum_cb, col_identity_cb, 0, 0, 0, 0, N, 1, N);
            recip_tile_init<false>();
            MATH((recip_tile<false>(0 /*dst_index*/, VectorMode::C)));
            tile_regs_commit();
            tile_regs_wait();
            configure_single_tile_pack(scratch_cb);
            pack_tile(0, scratch_cb);
            tile_regs_release();
            CircularBuffer(scratch_cb).push_back(1);
            CircularBuffer(cur_sum_cb).pop_front(1);
        }
        {
            // 3: out *= bcast_cols(1 / sum) (eltwise: in0 = out -> srcA, in1 = scratch -> srcB) -> the bf16 output CB
            constexpr uint32_t batch = (head_dim_t_ < dst_size) ? head_dim_t_ : dst_size;
            reconfig_data_format(cur_out_cb, scratch_cb);
            pack_reconfig_data_format(normalized_out_cb);
            mul_bcast_cols_init(cur_out_cb, scratch_cb);
            CircularBuffer(cur_out_cb).wait_front(head_dim_t_);
            CircularBuffer(scratch_cb).wait_front(1);
            CircularBuffer(normalized_out_cb).reserve_back(head_dim_t_);
            for (uint32_t base = 0; base < head_dim_t_; base += batch) {
                constexpr uint32_t last_batch = head_dim_t_ % batch;
                const uint32_t cur_batch = (base + batch <= head_dim_t_) ? batch : last_batch;
                tile_regs_acquire();
                for (uint32_t j = 0; j < cur_batch; ++j) {
                    mul_tiles_bcast_cols(cur_out_cb, scratch_cb, base + j, 0, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < cur_batch; ++j) {
                    pack_tile(j, normalized_out_cb);
                }
                tile_regs_release();
            }
            CircularBuffer(normalized_out_cb).push_back(head_dim_t_);
            CircularBuffer(scratch_cb).pop_front(1);
            CircularBuffer(cur_out_cb).pop_front(head_dim_t_);
        }
    }
}

}  // namespace sst
