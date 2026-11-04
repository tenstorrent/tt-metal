// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 14b model head, owner compute: the stream mean, this owner's partial sum of squares, the RMS scale and the
// norm weight (kernels/head/head_common.hpp). Every fp32 CB is unpacked straight to DST (UnpackToDestFp32, host).
// Compile-time args: INV_HIDDEN_BITS (1 / 4096), EPS_BITS (rms eps), QUARTER_BITS (0.25).
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/head/head_common.hpp"

using namespace head;

namespace {

constexpr uint32_t INV_HIDDEN_BITS = get_compile_time_arg_val(0);
constexpr uint32_t EPS_BITS = get_compile_time_arg_val(1);
constexpr uint32_t QUARTER_BITS = get_compile_time_arg_val(2);

FORCE_INLINE void load(uint32_t cb, uint32_t i, uint32_t d) {
    copy_init(cb);
    copy_tile(cb, i, d);
}

FORCE_INLINE void add_dst(uint32_t a, uint32_t b) {  // dst[a] += dst[b]
    add_binary_tile_init();
    add_binary_tile(a, b, a);
}

FORCE_INLINE void pack_to(uint32_t d, uint32_t cb) {
    pack_reconfig_data_format(cb);
    pack_tile(d, cb);
}

FORCE_INLINE void row_sum(uint32_t d) {  // column 0 of dst[d] <- the row sums of dst[d]
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(d);
}

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(CB_X, CB_M);
    CircularBuffer cb_x(CB_X), cb_w(CB_W), cb_m(CB_M), cb_p(CB_P), cb_g(CB_G), cb_out(CB_OUT);
    cb_x.wait_front(N_STREAMS * COLS);
    // m_j = 0.25 * (((x_0j + x_1j) + x_2j) + x_3j)
    cb_m.reserve_back(COLS);
    for (uint32_t j = 0; j < COLS; ++j) {
        tile_regs_acquire();
        load(CB_X, j, 0);
        for (uint32_t s = 1; s < N_STREAMS; ++s) {
            load(CB_X, s * COLS + j, 1);
            add_dst(0, 1);
        }
        binop_with_scalar_tile_init();
        mul_unary_tile(0, QUARTER_BITS);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, CB_M);
        tile_regs_release();
    }
    cb_m.push_back(COLS);
    cb_x.pop_front(N_STREAMS * COLS);
    // P = rowsum(m_0^2 + m_1^2)
    cb_m.wait_front(COLS);
    cb_p.reserve_back(1);
    tile_regs_acquire();
    load(CB_M, 0, 0);
    square_tile_init();
    square_tile(0);
    load(CB_M, 1, 1);
    square_tile_init();
    square_tile(1);
    add_dst(0, 1);
    row_sum(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_P);
    tile_regs_release();
    cb_p.push_back(1);
    // inv = rsqrt(rowsum(G0 + G1) / 4096 + eps); h_j = m_j * inv (column broadcast) * w_j (row broadcast)
    cb_g.wait_front(2);
    cb_w.wait_front(COLS);
    cb_out.reserve_back(COLS);
    for (uint32_t j = 0; j < COLS; ++j) {
        tile_regs_acquire();
        load(CB_G, 0, 2);
        load(CB_G, 1, 3);
        add_dst(2, 3);
        row_sum(2);
        binop_with_scalar_tile_init();
        mul_unary_tile(2, INV_HIDDEN_BITS);
        add_unary_tile(2, EPS_BITS);
        rsqrt_tile_init();
        rsqrt_tile(2);
        load(CB_M, j, 0);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 2);
        load(CB_W, j, 1);
        sfpu_mul_bcast_row_init();
        sfpu_mul_bcast_row(0, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, CB_OUT);
        tile_regs_release();
    }
    cb_out.push_back(COLS);
    cb_g.pop_front(2);
    cb_w.pop_front(COLS);
    cb_m.pop_front(COLS);
}
