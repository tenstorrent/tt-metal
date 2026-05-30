// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_binary.h"

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#ifdef SFPU_BINARY_OP
#include "api/compute/eltwise_unary/eltwise_unary.h"
#endif

// Metal 2.0 binary SFPU kernel (dataflow-buffer based). Copies both operands into DST[0]/DST[1] via
// copy_tile, runs the SFPU binary op (SFPU_OP_CHAIN_0), and packs DST[0]. Used for div_binary and
// add_int; add_int relies on fp32_dest_acc to promote Int8 inputs to sign-magnitude Int32 in dest.
void kernel_main() {
    const uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    const uint32_t per_core_block_size = get_arg(args::per_core_block_size);
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);

    binary_op_init_common(dfb_in0.get_id(), dfb_in1.get_id(), dfb_out.get_id());
#ifdef SFPU_OP_INIT_0
    SFPU_OP_INIT_0
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        dfb_in0.wait_front(per_core_block_size);
        dfb_in1.wait_front(per_core_block_size);
        dfb_out.reserve_back(per_core_block_size);

        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            acquire_dst();
            copy_tile_to_dst_init_short(dfb_in0.get_id());
            copy_tile(dfb_in0.get_id(), i, 0);
            copy_tile_to_dst_init_short(dfb_in1.get_id());
            copy_tile(dfb_in1.get_id(), i, 1);
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
            pack_tile(0, dfb_out.get_id());
            release_dst();
        }

        dfb_in0.pop_front(per_core_block_size);
        dfb_in1.pop_front(per_core_block_size);
        dfb_out.push_back(per_core_block_size);
    }
}
