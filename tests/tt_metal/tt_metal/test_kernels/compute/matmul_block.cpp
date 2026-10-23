// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t block_tile_dim = get_arg(args::block_tile_dim);
    constexpr uint32_t dst_tile_rows = get_arg(args::dst_tile_rows);
    constexpr uint32_t dst_tile_cols = get_arg(args::dst_tile_cols);
    constexpr uint32_t block_cnt = get_arg(args::block_cnt);
    constexpr uint32_t in0_block_tile_cnt = get_arg(args::in0_block_tile_cnt);
    constexpr uint32_t in1_block_tile_cnt = get_arg(args::in1_block_tile_cnt);
    constexpr uint32_t out_block_tile_cnt = get_arg(args::out_block_tile_cnt);

    DataflowBuffer dfb0(dfb::in0);
    DataflowBuffer dfb1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::in0, dfb::in1, dfb::out);

#if (TEST_INIT_SHORT == 1)
#if (WITH_DT == 1)
    // Intentionally wrong init (in0/in1 swapped, dst dims off-by-one) — exercises
    // the reconfig_data_format_srca + matmul_block_init ability to re-init data formats from a
    // bad state. The legacy variant of this test used a separate uint16 CB to model a true
    // data-format mismatch; with DFBs we reuse dfb_out as the old srcA operand since adding an
    // unused dummy DFB has no Metal 2.0 equivalent. The reconfig path is still exercised.
    matmul_block_init(dfb::in1, dfb::in0, false, dst_tile_cols - 1, dst_tile_rows - 1, block_tile_dim - 1);
    reconfig_data_format_srca(dfb::out, dfb::in1);
    matmul_block_init(dfb::in0, dfb::in1, false, dst_tile_cols, dst_tile_rows, block_tile_dim);
#elif (WITH_DT == 0)
    matmul_block_init(dfb::in1, dfb::in0, false, dst_tile_cols - 1, dst_tile_rows - 1, block_tile_dim - 1);
    matmul_block_init(dfb::in0, dfb::in1, false, dst_tile_cols, dst_tile_rows, block_tile_dim);
#endif
#elif (TEST_INIT_SHORT == 0)
    matmul_block_init(dfb::in0, dfb::in1, false, dst_tile_cols, dst_tile_rows, block_tile_dim);
#endif

    tile_regs_acquire();
    tile_regs_wait();
    for (uint32_t b = 0; b < block_cnt; ++b) {
        dfb0.wait_front(in0_block_tile_cnt);
        dfb1.wait_front(in1_block_tile_cnt);

        matmul_block(dfb::in0, dfb::in1, 0, 0, 0, false, dst_tile_cols, dst_tile_rows, block_tile_dim);

        dfb0.pop_front(in0_block_tile_cnt);
        dfb1.pop_front(in1_block_tile_cnt);
    }

    dfb_out.reserve_back(out_block_tile_cnt);
    for (uint32_t i = 0; i < out_block_tile_cnt; ++i) {
        pack_tile(i, dfb::out);
    }
    dfb_out.push_back(out_block_tile_cnt);
    tile_regs_commit();
    tile_regs_release();
}
