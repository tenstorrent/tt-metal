// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"  // legacy matmul_block_init / matmul_block, compute_kernel_hw_startup, pack_tile
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

// Legacy (id-based) block matmul kernel, classic circular buffers. Baseline for
// TensixMatmulBlockSpecMatchesLegacy: it must produce output bit-for-bit identical to matmul_block_2_0.cpp on
// the same inputs. Single block (block_cnt == 1): C = A*B where A is rt_dim x kt_dim tiles (c_0), B is
// kt_dim x ct_dim tiles (c_1), and C is rt_dim x ct_dim tiles (c_16). Compile args are the runtime block dims
// {ct_dim, rt_dim, kt_dim}. hw_startup + pack are legacy CB-id in BOTH kernels; only the matmul_block call
// differs (legacy here, experimental:: in the 2_0 kernel) to isolate matmul_block.
void kernel_main() {
    constexpr std::uint32_t ct_dim = get_compile_time_arg_val(0);
    constexpr std::uint32_t rt_dim = get_compile_time_arg_val(1);
    constexpr std::uint32_t kt_dim = get_compile_time_arg_val(2);
    constexpr std::uint32_t in0_block_tile_cnt = rt_dim * kt_dim;  // A block
    constexpr std::uint32_t in1_block_tile_cnt = kt_dim * ct_dim;  // B block
    constexpr std::uint32_t out_block_tile_cnt = rt_dim * ct_dim;  // C block

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb1(tt::CBIndex::c_1);
    CircularBuffer cb16(tt::CBIndex::c_16);

    compute_kernel_hw_startup<SrcOrder::Reverse>(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
    matmul_block_init(tt::CBIndex::c_0, tt::CBIndex::c_1, false, ct_dim, rt_dim, kt_dim);

    tile_regs_acquire();
    cb0.wait_front(in0_block_tile_cnt);
    cb1.wait_front(in1_block_tile_cnt);
    matmul_block(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0, false, ct_dim, rt_dim, kt_dim);
    cb0.pop_front(in0_block_tile_cnt);
    cb1.pop_front(in1_block_tile_cnt);
    tile_regs_commit();
    tile_regs_wait();

    // Pack out (legacy CB-id in both kernels).
    cb16.reserve_back(out_block_tile_cnt);
    for (std::uint32_t i = 0; i < out_block_tile_cnt; ++i) {
        pack_tile(i, tt::CBIndex::c_16);
    }
    cb16.push_back(out_block_tile_cnt);

    tile_regs_release();
}
