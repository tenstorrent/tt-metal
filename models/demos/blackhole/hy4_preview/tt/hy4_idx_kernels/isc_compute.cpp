// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 decode indexer score, compute. Per batch of kc key row tiles:
//   phase A: for each query row r: qk[r][col] = relu(Q_r (32 heads x 128) @ K_col^T)  (4 matmul tile ops per key tile, DST accumulation, relu in the packer) -> CB_QK
//   phase B: out[col] = sum_r Wmask_r @ qk[r][col]  (one matmul tile op per (r, col); Wmask_r has only row r non-zero, so out row r = sum_h w[r,h] qk[r][col][h,:]) -> CB_OUT
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

using namespace ckernel;

constexpr uint32_t S = get_compile_time_arg_val(0);
constexpr uint32_t KC = get_compile_time_arg_val(1);
constexpr uint32_t CB_Q = get_compile_time_arg_val(2);
constexpr uint32_t CB_W = get_compile_time_arg_val(3);
constexpr uint32_t CB_K = get_compile_time_arg_val(4);
constexpr uint32_t CB_QK = get_compile_time_arg_val(5);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(6);
constexpr uint32_t DT = 4;  // head_dim 128 = 4 tiles

void kernel_main() {
    const uint32_t nrows = get_arg_val<uint32_t>(0);
    CircularBuffer cq(CB_Q), cw(CB_W), ck(CB_K), cqk(CB_QK), co(CB_OUT);
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_Q, CB_K, CB_QK);
    cq.wait_front(S * DT);
    cw.wait_front(S);
    const uint32_t nb = (nrows + KC - 1) / KC;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint32_t kc = (b * KC + KC <= nrows) ? KC : (nrows - b * KC);
        ck.wait_front(kc * DT);
        cqk.reserve_back(S * kc);
        // ---- phase A
        reconfig_data_format(CB_QK, CB_K, CB_W, CB_Q);  // srcA <- k (bfp8), srcB <- q
        matmul_block_init(CB_Q, CB_K, 1 /*transpose k*/, 1 /*ct*/, 1 /*rt*/, DT /*kt*/);
        pack_relu_config(ReluConfig::zero());
        for (uint32_t r = 0; r < S; ++r) {
            tile_regs_acquire();
            for (uint32_t col = 0; col < kc; ++col) {
                for (uint32_t d = 0; d < DT; ++d) {
                    matmul_block(CB_Q, CB_K, r * DT + d, col * DT + d, col, 1, 1, 1, DT);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t col = 0; col < kc; ++col) {
                pack_tile<true>(col, CB_QK, r * kc + col);
            }
            tile_regs_release();
        }
        cqk.push_back(S * kc);
        ck.pop_front(kc * DT);
        // ---- phase B
        cqk.wait_front(S * kc);
        co.reserve_back(kc);
        pack_relu_config(ReluConfig::none());
        reconfig_data_format(CB_K, CB_QK, CB_Q, CB_W);  // srcA <- qk (in1), srcB <- w (in0)
        matmul_block_init(CB_W, CB_QK, 0 /*no transpose*/, 1, 1, 1);
        tile_regs_acquire();
        for (uint32_t col = 0; col < kc; ++col) {
            for (uint32_t r = 0; r < S; ++r) {
                matmul_block(CB_W, CB_QK, r, r * kc + col, col, 0, 1, 1, 1);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t col = 0; col < kc; ++col) {
            pack_tile(col, CB_OUT);
        }
        tile_regs_release();
        co.push_back(kc);
        cqk.pop_front(S * kc);
    }
}
