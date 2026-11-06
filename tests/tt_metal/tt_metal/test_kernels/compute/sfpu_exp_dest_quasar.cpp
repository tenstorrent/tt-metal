// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Example: exp on Dest through the SFPU op-type compute API. The unary loop of eltwise_sfpu_2_0.cpp with
// exp_tile_init(); exp_tile(0); replaced by sfpu_init / sfpu_tile on Dest.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/experimental/sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    const uint32_t per_core_block_size = get_arg(args::per_core_block_size);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);
    constexpr uint32_t in_id = dfb::in;  // the DFB id is a compile-time constant: it names the tiles' format
    const uint32_t out_id = dfb_out.get_id();

    compute_kernel_hw_startup(in_id, out_id);
    copy_init(in_id);
    sfpu_init<op::Exp<>, SfpuReg::Dest, in_id>();

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            dfb_in.wait_front(1);
            dfb_out.reserve_back(1);

            tile_regs_acquire();
            copy_init(in_id);
            copy_tile(in_id, /*tile_index=*/0, /*dst_index=*/0);
            sfpu_tile<op::Exp<>, SfpuReg::Dest, in_id>(0);  // exp(Dest[0]) in place
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(/*dst_index=*/0, out_id);
            tile_regs_release();

            dfb_out.push_back(1);
            dfb_in.pop_front(1);
        }
    }
}
