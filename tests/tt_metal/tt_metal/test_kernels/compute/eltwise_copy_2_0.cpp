// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 host-API version of eltwise_copy.cpp. Uses named compile/runtime
// args and DataflowBuffer endpoint accessors. The legacy CircularBuffer path
// remains in eltwise_copy.cpp for callers still on the legacy host API.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_arg(args::per_core_tile_cnt);

    unary_op_init_common(0, 1);

#ifdef PACK_RELU
    pack_relu_config(ReluConfig::from_packed(get_arg(args::relu_config)));
#endif

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);
    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        tile_regs_acquire();
        tile_regs_wait();

        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);
        copy_tile(dfb::in, 0, 0);
        pack_tile(0, dfb::out);
        dfb_in.pop_front(1);
        dfb_out.push_back(1);

        tile_regs_commit();
        tile_regs_release();
    }
}
